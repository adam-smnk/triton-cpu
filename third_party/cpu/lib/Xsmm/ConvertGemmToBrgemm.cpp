//===- ConvertGemmToBrgemm.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "cpu/include/Xsmm/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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

// Collapse whole reduction loop with a GEMM into equivalent BRGEMM operation.
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
//   %res0 = BRGEMM %A, %B, %init_val
//   %res1 = tt.advance %A, [0, ((%ub - %lb) / %step)]
//   %res2 = tt.advance %B, [((%ub - %lb) / %step), 0]
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
    BlockArgument accArg = dyn_cast<BlockArgument>(acc);
    if (!forOp || !accArg)
      return rewriter.notifyMatchFailure(dotOp, "not a reduction loop");
    auto iterArgs = llvm::to_vector(forOp.getInitArgs());
    // TODO: Relax this check. It is needed to collapse whole loop but
    //       alternatively only BRGEMM could be pulled out.
    if (iterArgs.size() != 3)
      return rewriter.notifyMatchFailure(dotOp, "invalid number of iter_args");

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

    // Check for pointer increments and validate their steps.
    // Each input is expected to advance only in its reduction dimension.
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

    // Create new block pointers spanning the whole reduction dimension.
    int64_t numTiles = (*loopUB - *loopLB) / (*loopStep);
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

    // Load the new tensors.
    // Only the result shape is updated, all the source metadata remains
    // unchanged.
    auto matA = rewriter.create<triton::LoadOp>(
        loc, newLhsPtr, loadMatA.getBoundaryCheck(), loadMatA.getPadding(),
        loadMatA.getCache(), loadMatA.getEvict(), loadMatA.getIsVolatile());
    auto matB = rewriter.create<triton::LoadOp>(
        loc, newRhsPtr, loadMatB.getBoundaryCheck(), loadMatB.getPadding(),
        loadMatB.getCache(), loadMatB.getEvict(), loadMatB.getIsVolatile());

    auto vecTyA =
        VectorType::get(lhsResType.getShape(), lhsResType.getElementType());
    auto castA = rewriter.create<UnrealizedConversionCastOp>(loc, vecTyA,
                                                             ValueRange{matA});
    auto vecTyB =
        VectorType::get(rhsResType.getShape(), rhsResType.getElementType());
    auto castB = rewriter.create<UnrealizedConversionCastOp>(loc, vecTyB,
                                                             ValueRange{matB});
    auto vecTyAcc = VectorType::get(acc.getType().getShape(),
                                    acc.getType().getElementType());
    unsigned accIdx = accArg.getArgNumber() - numIvs;
    auto castAcc = rewriter.create<UnrealizedConversionCastOp>(
        loc, vecTyAcc, ValueRange{iterArgs[accIdx]});

    // Split reduction dimension into tiles.
    // The number of tiles represents the batch dimension.
    auto expandedTypeA = VectorType::get(
        {resShape[0], numTiles, *lhsStepReduction}, vecTyA.getElementType());
    auto expandedTypeB = VectorType::get(
        {numTiles, *rhsStepReduction, resShape[1]}, vecTyB.getElementType());
    auto expandA = rewriter.create<vector::ShapeCastOp>(loc, expandedTypeA,
                                                        castA.getOutputs()[0]);
    auto expandB = rewriter.create<vector::ShapeCastOp>(loc, expandedTypeB,
                                                        castB.getOutputs()[0]);

    // Construct BRGEMM operation.
    // Generic is used as matrix A lacks transposition to match linalg named op.
    // TODO: Should the input operands be normalized with tranposes?
    //
    // Linalg operations would be more suitable to represent BRGEMM on tensors.
    // However, upstream vectorization does not compose with triton's tensor
    // semantics and further triton-cpu conversion to vector ops.
    // Ideally, this should be handled by additional conversion patterns that
    // map linalg ops to vector. For simplicity, the conversion is done
    // immediately here which also ties this rewrite to triton-to-triton-cpu
    // conversion.
    // TODO: Revisit lowering strategy - consider having a separate conversions
    //       for linalg ops and tensor casts, or at least use type converter
    //       instead of manual casts.
    auto mapA = AffineMap::getMultiDimMapWithTargets(4, {1, 0, 3}, ctx);
    auto mapB = AffineMap::getMultiDimMapWithTargets(4, {0, 3, 2}, ctx);
    auto mapC = AffineMap::getMultiDimMapWithTargets(4, {1, 2}, ctx);
    SmallVector<vector::IteratorType> iteratorTypes{
        vector::IteratorType::reduction, vector::IteratorType::parallel,
        vector::IteratorType::parallel, vector::IteratorType::reduction};
    auto brgemmOp = rewriter.create<vector::ContractionOp>(
        loc, expandA.getResult(), expandB.getResult(), castAcc.getOutputs()[0],
        rewriter.getAffineMapArrayAttr({mapA, mapB, mapC}),
        rewriter.getArrayAttr(llvm::to_vector(llvm::map_range(
            iteratorTypes, [&](vector::IteratorType itTy) -> mlir::Attribute {
              return vector::IteratorTypeAttr::get(ctx, itTy);
            }))));
    auto brgemmCast = rewriter.create<UnrealizedConversionCastOp>(
        loc, acc.getType(), ValueRange{brgemmOp.getResult()});

    // Increment the base pointers such that the whole loop can be removed.
    // TODO: Revisit this part.
    //       Only the BRGEMM could be pulled out of the loop and the rest
    //       could be left as is.
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value reductionStepConst =
        rewriter.create<arith::ConstantIndexOp>(loc, *lhsStepReduction);
    Value numIterConst = rewriter.create<arith::ConstantIndexOp>(loc, numTiles);
    Value reductionOffset =
        rewriter.create<arith::MulIOp>(loc, reductionStepConst, numIterConst);
    auto advanceA = rewriter.create<triton::AdvanceOp>(
        loc, lhsBlockPtr.getResult().getType(), lhsBlockPtr,
        ValueRange{zero, reductionOffset});
    auto advanceB = rewriter.create<triton::AdvanceOp>(
        loc, rhsBlockPtr.getResult().getType(), rhsBlockPtr,
        ValueRange{reductionOffset, zero});

    rewriter.replaceOp(forOp,
                       ValueRange{brgemmCast.getOutputs()[0],
                                  advanceA.getResult(), advanceB.getResult()});

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
