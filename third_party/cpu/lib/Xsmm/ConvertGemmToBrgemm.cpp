//===- ConvertGemmToBrgemm.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "cpu/include/Xsmm/Passes.h"

#include "ValueUtils.h"
#include "VnniUtils.h"
#include "XsmmUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include <algorithm>
#include <iterator>
#include <optional>
#include <utility>

using namespace mlir;
using namespace mlir::vector;
using namespace mlir::func;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace mlir {
namespace triton {
namespace cpu {
#define GEN_PASS_DEF_CONVERTGEMMTOBRGEMM
#include "cpu/include/Xsmm/Passes.h.inc"
} // namespace cpu
} // namespace triton
} // namespace mlir

namespace {

// Rewrites the following pattern:
//   %0 = tt.make_tensor_ptr %base_ptr0 : tensor<MxK>
//   %1 = tt.make_tensor_ptr %base_ptr1 : tensor<KxN>
//   %2:3 = scf.for %arg3 = %lb to %ub step %step
//       iter_args(%acc = %init_val, %ptr_A = %0, %ptr_B = %1)
//     %A = tt.load %ptr_A
//     %B = tt.load %ptr_B
//     %res = tt.dot %A, %B, %acc
//     %ptr_A_next = tt.advance %ptr_A, [0, %stepK]
//     %ptr_B_next = tt.advance %ptr_B, [%stepK, %0]
//     scf.yield %res, %ptr_A_next, %ptr_V_next
// into:
//   %A = tt.make_tensor_ptr %base_ptr0 : tensor<M x TILES x k>
//   %B = tt.make_tensor_ptr %base_ptr1 : tensor<TILES x k x N>
//   %res = BRGEMM %A, %B, %init_val
struct DotReductionLoopToBrgemm : public OpRewritePattern<triton::DotOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::DotOp dotOp,
                                PatternRewriter &rewriter) const override {
    Location loc = dotOp.getLoc();
    MLIRContext *ctx = dotOp.getContext();

    // Check if there is any loop around the contraction and if the accumulation
    // value comes from loop's arguments.
    TypedValue<RankedTensorType> acc = dotOp.getC();
    if (acc.getType().getRank() != 2)
      return rewriter.notifyMatchFailure(dotOp, "expects 2D GEMM");

    auto forOp = dyn_cast<scf::ForOp>(dotOp->getParentOp());
    BlockArgument blockArg = dyn_cast<BlockArgument>(acc);
    if (!forOp || !blockArg)
      return rewriter.notifyMatchFailure(dotOp, "not a reduction loop");

    // TODO: Allow dynamic ranges.
    auto loopUB = getConstantIntValue(forOp.getUpperBound());
    auto loopLB = getConstantIntValue(forOp.getLowerBound());
    auto loopStep = getConstantIntValue(forOp.getStep());
    if (!loopUB || !loopLB || !loopStep)
      return rewriter.notifyMatchFailure(dotOp,
                                         "expects loop with static range");

    // Assume that the loop's range and all pointer advances are known
    // statically. Thus, the induction variable should be unused.
    Value loopIv = forOp.getInductionVar();
    if (!loopIv.use_empty())
      return rewriter.notifyMatchFailure(dotOp,
                                         "expects unused induction variable");

    // The subgraph should a simple reduction loop containing a GEMM operation.
    // Validate presence of the following chain:
    //   iter_arg -> contraction -> yield
    // and that there are no other users.
    TypedValue<RankedTensorType> res = dotOp.getD();
    if (!acc.hasOneUse() || !res.hasOneUse() ||
        !isa<scf::YieldOp>(*res.getUsers().begin()))
      return rewriter.notifyMatchFailure(dotOp, "GEMM subgraph does not match");

    auto loadMatA = dotOp.getA().getDefiningOp<triton::LoadOp>();
    auto loadMatB = dotOp.getB().getDefiningOp<triton::LoadOp>();
    if (!loadMatA || !loadMatB)
      return rewriter.notifyMatchFailure(dotOp, "expect GEMM input loads");

    // Constrain input pointers to the following subgraph:
    //   iter_arg -> (load, increment) -> yield
    BlockArgument lhsArg = dyn_cast<BlockArgument>(loadMatA.getPtr());
    BlockArgument rhsArg = dyn_cast<BlockArgument>(loadMatB.getPtr());
    if (!lhsArg || std::distance(lhsArg.use_begin(), lhsArg.use_end()) != 2 ||
        !rhsArg || std::distance(rhsArg.use_begin(), rhsArg.use_end()) != 2)
      return rewriter.notifyMatchFailure(dotOp, "expect iter_args pointers");

    auto iterArgs = llvm::to_vector(forOp.getInitArgs());
    auto numIvs = forOp.getNumInductionVars();

    // Input sources should be block pointers.
    // TODO: Account for transposed GEMM operands.
    auto lhsBlockPtr = dyn_cast_or_null<triton::MakeTensorPtrOp>(
        iterArgs[lhsArg.getArgNumber() - numIvs].getDefiningOp());
    auto rhsBlockPtr = dyn_cast_or_null<triton::MakeTensorPtrOp>(
        iterArgs[rhsArg.getArgNumber() - numIvs].getDefiningOp());
    if (!lhsBlockPtr || lhsBlockPtr.getOrder() != ArrayRef<int32_t>{1, 0} ||
        !rhsBlockPtr || rhsBlockPtr.getOrder() != ArrayRef<int32_t>{1, 0})
      return rewriter.notifyMatchFailure(dotOp, "expected block pointers");

    auto yields = *forOp.getYieldedValuesMutable();
    auto lhsAdvanceOp = yields[lhsArg.getArgNumber() - numIvs]
                            .get()
                            .getDefiningOp<triton::AdvanceOp>();
    auto rhsAdvanceOp = yields[rhsArg.getArgNumber() - numIvs]
                            .get()
                            .getDefiningOp<triton::AdvanceOp>();
    if (!lhsAdvanceOp || !rhsAdvanceOp)
      return rewriter.notifyMatchFailure(dotOp, "expected ptr advance");

    auto resShape = res.getType().getShape();
    auto lhsPtrOffsets = lhsAdvanceOp.getOffsets();
    auto lhsStepParallel = getConstantIntValue(lhsPtrOffsets[0]);
    auto lhsStepReduction = getConstantIntValue(lhsPtrOffsets[1]);
    if (!lhsStepParallel || *lhsStepParallel != 0 || !lhsStepReduction ||
        *lhsStepReduction != resShape[1])
      return rewriter.notifyMatchFailure(dotOp, "invalid lhs increments");

    auto rhsPtrOffsets = rhsAdvanceOp.getOffsets();
    auto rhsStepReduction = getConstantIntValue(rhsPtrOffsets[0]);
    auto rhsStepParallel = getConstantIntValue(rhsPtrOffsets[1]);
    if (!rhsStepReduction || *rhsStepReduction != *lhsStepReduction ||
        !rhsStepParallel || *rhsStepParallel != 0)
      return rewriter.notifyMatchFailure(dotOp, "invalid rhs increments");

    // Collapse the loop and create equivalent BRGEMM operation.
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointAfter(forOp);

    int64_t numTiles = (*loopUB - *loopLB) * (*loopStep);
    auto lhsResType =
        RankedTensorType::get({resShape[0], (*lhsStepReduction) * numTiles},
                              res.getType().getElementType());
    auto newLhsPtr = rewriter.create<triton::MakeTensorPtrOp>(
        loc, PointerType::get(lhsResType, 1), lhsBlockPtr.getBase(),
        lhsBlockPtr.getShape(), lhsBlockPtr.getStrides(),
        lhsBlockPtr.getOffsets(), lhsBlockPtr.getOrderAttr());
    auto rhsResType =
        RankedTensorType::get({(*rhsStepReduction) * numTiles, resShape[1]},
                              res.getType().getElementType());
    auto newRhsPtr = rewriter.create<triton::MakeTensorPtrOp>(
        loc, PointerType::get(rhsResType, 1), rhsBlockPtr.getBase(),
        rhsBlockPtr.getShape(), rhsBlockPtr.getStrides(),
        rhsBlockPtr.getOffsets(), rhsBlockPtr.getOrderAttr());

    auto matA = rewriter.create<triton::LoadOp>(
        loc, newLhsPtr, loadMatA.getBoundaryCheck(), loadMatA.getPadding(),
        loadMatA.getCache(), loadMatA.getEvict(), loadMatA.getIsVolatile());
    auto matB = rewriter.create<triton::LoadOp>(
        loc, newRhsPtr, loadMatB.getBoundaryCheck(), loadMatB.getPadding(),
        loadMatB.getCache(), loadMatB.getEvict(), loadMatB.getIsVolatile());

    // SmallVector<int64_t> lhsVectorShape{lhsTensorShape.begin(),
    //                                     lhsTensorShape.end()};
    // auto vecA = rewriter.create<UnrealizedConversionCastOp>(
    //     loc, VectorType::get(lhsVectorShape, res.getType().getElementType()),
    //     ValueRange{matA});
    // SmallVector<int64_t> rhsVectorShape{rhsTensorShape.begin(),
    //                                     rhsTensorShape.end()};
    // auto vecB = rewriter.create<UnrealizedConversionCastOp>(
    //     loc, VectorType::get(rhsVectorShape, res.getType().getElementType()),
    //     ValueRange{matB});

    return success();
  }
};

struct ConvertGemmToBrgemm
    : public triton::cpu::impl::ConvertGemmToBrgemmBase<ConvertGemmToBrgemm> {
  using ConvertGemmToBrgemmBase::ConvertGemmToBrgemmBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<DotReductionLoopToBrgemm>(context);
    if (failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                  std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace
