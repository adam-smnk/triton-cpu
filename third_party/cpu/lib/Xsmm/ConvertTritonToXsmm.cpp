//===- ConvertTritonToXsmm.cpp ----------------------------------*- C++ -*-===//
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
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"

#include "cpu/include/Analysis/TensorPtrShapeInfo.h"

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
#define GEN_PASS_DEF_CONVERTTRITONTOXSMM
#include "cpu/include/Xsmm/Passes.h.inc"
} // namespace cpu
} // namespace triton
} // namespace mlir

namespace {

// Helper from MemoryOpConversion.
// Extract memref out of block pointer.
static Value extractMemRef(PatternRewriter &rewriter, Value ptr,
                           ModuleTensorPtrShapeInfoAnalysis &shapeAnalysis) {
  Location loc = ptr.getLoc();
  MLIRContext *ctx = ptr.getContext();

  auto tensorTy = dyn_cast<RankedTensorType>(
      dyn_cast<PointerType>(ptr.getType()).getPointeeType());
  auto elemTy = tensorTy.getElementType();
  auto shapeInfo = shapeAnalysis.getPtrShapeInfo(ptr);
  Type memRefTy;
  if (shapeInfo && shapeInfo->getRank() > 0) {
    auto layout = StridedLayoutAttr::get(ctx, 0, shapeInfo->getStrides());
    memRefTy = MemRefType::get(shapeInfo->getShape(), elemTy, layout);
  } else {
    SmallVector<int64_t> dynVals(tensorTy.getRank(), ShapedType::kDynamic);
    auto layout = StridedLayoutAttr::get(ctx, 0, dynVals);
    memRefTy = MemRefType::get(dynVals, elemTy, layout);
  }
  return rewriter.create<triton::cpu::ExtractMemRefOp>(loc, memRefTy, ptr);
}

static Value getMemrefSource(PatternRewriter &rewriter, Operation *op,
                             TypedValue<RankedTensorType> operand,
                             ModuleTensorPtrShapeInfoAnalysis &shapeAnalysis) {
  Location loc = op->getLoc();
  MLIRContext *ctx = op->getContext();

  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(op);

  RankedTensorType tensorTy = operand.getType();

  if (auto loadOp = dyn_cast_or_null<triton::LoadOp>(operand.getDefiningOp())) {
    auto ptr = loadOp.getPtr();
    if (triton::isTensorPointerType(ptr.getType())) {
      auto memref = extractMemRef(rewriter, ptr, shapeAnalysis);
      auto indices =
          rewriter.create<triton::cpu::ExtractIndicesOp>(loc, ptr).getResults();
      SmallVector<int64_t> strides(tensorTy.getRank(), 1);

      return rewriter.create<memref::SubViewOp>(
          loc, memref, getAsOpFoldResult(indices),
          getAsIndexOpFoldResult(ctx, tensorTy.getShape()),
          getAsIndexOpFoldResult(ctx, strides));
    }
  }

  MemRefType memTy =
      MemRefType::get(tensorTy.getShape(), tensorTy.getElementType());
  auto alloca = rewriter.create<memref::AllocaOp>(loc, memTy);
  rewriter.create<triton::cpu::StoreOp>(loc, operand, alloca);

  return alloca;
}

// Helper to move accumulation buffer outside of GEMM reduction loop.
// Returns new accumulation buffer or std::nullopt, otherwise.
//
// Rewrites the following pattern:
//   %acc = ... tensor<...>
//   %0 = scf.for ... iter_args(%acc)
//     %res = GEMM(%A, %B, %acc) -> tensor<...>
//     scf.yield %res
//   consumer(%0)
// into:
//   %hoisted = ... memref<...>
//   %unused = %scf.for ... iter_args(%acc)
//     %res = GEMM(%A, %B, %acc)
//     scf.yield %acc
//   %0 = load(%hoisted) -> tensor<...>
//   consumer(%0)
//
// This rewrite should be used as a part of contraction to memref conversion.
static std::optional<Value>
hoistAccumulationBuffer(PatternRewriter &rewriter, Operation *op,
                        TypedValue<RankedTensorType> operand,
                        ModuleTensorPtrShapeInfoAnalysis &shapeAnalysis) {
  Location loc = op->getLoc();

  // Check if there is any loop around the contraction and if the operand
  // comes from loop's arguments.
  auto forOp = dyn_cast<scf::ForOp>(op->getParentOp());
  BlockArgument blockArg = dyn_cast<BlockArgument>(operand);
  if (!forOp || !blockArg)
    return std::nullopt;
  OpOperand *loopArg = forOp.getTiedLoopInit(blockArg);
  if (!loopArg)
    return std::nullopt;

  // The accumulation iter_arg can be safely moved outside the loop only
  // for the following chain: iter_arg -> contraction -> yield
  // and there are no other users.
  Value res = op->getResults()[0];
  if (!operand.hasOneUse() || !res.hasOneUse() ||
      !isa<scf::YieldOp>(*res.getUsers().begin()))
    return std::nullopt;

  // Create a buffer outside the loop.
  Value accBuf = getMemrefSource(
      rewriter, forOp, dyn_cast<TypedValue<RankedTensorType>>(loopArg->get()),
      shapeAnalysis);

  // For simplicity, feed the iter_arg directly into loop yield terminator.
  // Canonicalizer will folded them away later.
  rewriter.replaceAllUsesWith(res, operand);

  // Replace the corresponding loop result with the latest value read from the
  // accumulation buffer.
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointAfter(forOp);

  auto loadOp =
      rewriter.create<triton::cpu::LoadOp>(loc, operand.getType(), accBuf);
  rewriter.replaceAllUsesWith(forOp.getTiedLoopResult(blockArg),
                              loadOp.getResult());

  return accBuf;
}

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

  DotReductionLoopToBrgemm(MLIRContext *context,
                           ModuleTensorPtrShapeInfoAnalysis &shapeInfoAnalysis,
                           PatternBenefit benefit = 10)
      : OpRewritePattern<triton::DotOp>(context, benefit),
        shapeAnalysis(shapeInfoAnalysis) {}

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
    BlockArgument accBbArg = dyn_cast<BlockArgument>(acc);
    if (!forOp || !accBbArg)
      return rewriter.notifyMatchFailure(dotOp, "not a reduction loop");
    OpOperand *accArg = forOp.getTiedLoopInit(accBbArg);
    if (!accArg)
      return rewriter.notifyMatchFailure(
          dotOp, "expects iter_args accumulation value");
    // TODO: Relax this check. It is needed to collapse whole loop but
    //       alternatively only BRGEMM could be pulled out.
    if (forOp.getNumRegionIterArgs() != 3)
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
    BlockArgument lhsBbArg = dyn_cast<BlockArgument>(loadMatA.getPtr());
    BlockArgument rhsBbArg = dyn_cast<BlockArgument>(loadMatB.getPtr());
    if (!lhsBbArg || !rhsBbArg)
      return rewriter.notifyMatchFailure(dotOp, "expect block arg pointers");
    OpOperand *lhsArg = forOp.getTiedLoopInit(lhsBbArg);
    OpOperand *rhsArg = forOp.getTiedLoopInit(rhsBbArg);
    if (!lhsArg ||
        std::distance(lhsBbArg.use_begin(), lhsBbArg.use_end()) != 2 ||
        !rhsArg || std::distance(rhsBbArg.use_begin(), rhsBbArg.use_end()) != 2)
      return rewriter.notifyMatchFailure(dotOp, "expect iter_args pointers");

    // Input sources should be block pointers.
    // TODO: Account for transposed GEMM operands.
    auto lhsBlockPtr = dyn_cast_or_null<triton::MakeTensorPtrOp>(
        lhsArg->get().getDefiningOp());
    auto rhsBlockPtr = dyn_cast_or_null<triton::MakeTensorPtrOp>(
        rhsArg->get().getDefiningOp());
    if (!lhsBlockPtr || lhsBlockPtr.getOrder() != ArrayRef<int32_t>{1, 0} ||
        !rhsBlockPtr || rhsBlockPtr.getOrder() != ArrayRef<int32_t>{1, 0})
      return rewriter.notifyMatchFailure(dotOp, "expected block pointers");

    // Check for pointer increments and validate their steps.
    // Each input is expected to advance only in its reduction dimension.
    auto lhsAdvanceOp = forOp.getTiedLoopYieldedValue(lhsBbArg)
                            ->get()
                            .getDefiningOp<triton::AdvanceOp>();
    auto rhsAdvanceOp = forOp.getTiedLoopYieldedValue(rhsBbArg)
                            ->get()
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
    rewriter.setInsertionPoint(forOp);

    // Create new block pointers spanning the whole reduction dimension.
    int64_t numTiles = (*loopUB - *loopLB) / (*loopStep);
    SmallVector<int64_t> lhsResShape{resShape[0],
                                     (*lhsStepReduction) * numTiles};
    SmallVector<int64_t> rhsResShape{(*rhsStepReduction) * numTiles,
                                     resShape[1]};

    auto lhsResType =
        RankedTensorType::get(lhsResShape, res.getType().getElementType());
    auto rhsResType =
        RankedTensorType::get(rhsResShape, res.getType().getElementType());
    auto newLhsPtr = rewriter.create<triton::MakeTensorPtrOp>(
        loc, PointerType::get(lhsResType, 1), lhsBlockPtr.getBase(),
        lhsBlockPtr.getShape(), lhsBlockPtr.getStrides(),
        lhsBlockPtr.getOffsets(), lhsBlockPtr.getOrderAttr());
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

    Value lhsBuf = getMemrefSource(
        rewriter, forOp,
        dyn_cast<TypedValue<RankedTensorType>>(matA.getResult()),
        shapeAnalysis);
    Value rhsBuf = getMemrefSource(
        rewriter, forOp,
        dyn_cast<TypedValue<RankedTensorType>>(matB.getResult()),
        shapeAnalysis);
    Value accBuf =
        getMemrefSource(rewriter, forOp,
                        dyn_cast<TypedValue<RankedTensorType>>(
                            accArg->get().getDefiningOp()->getResult(0)),
                        shapeAnalysis);

    // Split reduction dimension into tiles.
    // The number of tiles represents the batch dimension.
    auto expandA = rewriter.create<memref::ExpandShapeOp>(
        loc, SmallVector<int64_t>{resShape[0], numTiles, *lhsStepReduction},
        lhsBuf, SmallVector<ReassociationIndices>{{0}, {1, 2}});
    auto expandB = rewriter.create<memref::ExpandShapeOp>(
        loc, SmallVector<int64_t>{numTiles, *rhsStepReduction, resShape[1]},
        rhsBuf, SmallVector<ReassociationIndices>{{0, 1}, {2}});

    // Construct a temporary BRGEMM operation - used for XSMM call generation.
    // Generic is used as matrix A lacks transposition to match linalg named op.
    // TODO: Generalize XSMM utils to not require a concrete operation.
    SmallVector<Value> inputs{expandA, expandB, accBuf};
    auto mapA = AffineMap::getMultiDimMapWithTargets(4, {1, 0, 3}, ctx);
    auto mapB = AffineMap::getMultiDimMapWithTargets(4, {0, 3, 2}, ctx);
    auto mapC = AffineMap::getMultiDimMapWithTargets(4, {1, 2}, ctx);
    SmallVector<AffineMap> indexingMaps{mapA, mapB, mapC};
    auto brgemmOp = rewriter.create<linalg::GenericOp>(
        loc, /*ins=*/inputs, /*outs=*/ValueRange{}, indexingMaps,
        ArrayRef<mlir::utils::IteratorType>{
            mlir::utils::IteratorType::reduction,
            mlir::utils::IteratorType::parallel,
            mlir::utils::IteratorType::parallel,
            mlir::utils::IteratorType::reduction},
        /*doc=*/"", /*libraryCall=*/"",
        [](OpBuilder &builder, Location loc, ValueRange args) {
          Value res;
          if (isa<FloatType>(args[2].getType())) {
            Value mul = builder.create<arith::MulFOp>(loc, args[0], args[1]);
            res = builder.create<arith::AddFOp>(loc, args[2], mul);
          } else {
            Value mul = builder.create<arith::MulIOp>(loc, args[0], args[1]);
            res = builder.create<arith::AddIOp>(loc, args[2], mul);
          }
          builder.create<linalg::YieldOp>(loc, res);
        });

    SmallVector<Value> outputs{nullptr};
    auto brgemmInfo = xsmm::utils::isMappableToBrgemm(
        rewriter, brgemmOp, inputs, outputs, indexingMaps);
    assert(succeeded(brgemmInfo) && "failed mapping to XSMM");
    assert(!brgemmInfo->isVnni && "unhandled VNNI lowering");

    SmallVector<Attribute> flags;
    auto xsmmFuncs = xsmm::utils::buildBrgemmCalls(rewriter, brgemmOp, inputs,
                                                   *brgemmInfo, flags);
    // Cleanup the temporary op.
    rewriter.eraseOp(brgemmOp);

    // Load back the result to bring it back to tensor semantics.
    auto loadOp =
        rewriter.create<triton::cpu::LoadOp>(loc, res.getType(), accBuf);

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
                       ValueRange{loadOp.getResult(), advanceA.getResult(),
                                  advanceB.getResult()});

    return success();
  }

private:
  ModuleTensorPtrShapeInfoAnalysis &shapeAnalysis;
};

struct DotToXsmm : public OpRewritePattern<triton::DotOp> {
  using OpRewritePattern::OpRewritePattern;

  DotToXsmm(MLIRContext *ctx,
            ModuleTensorPtrShapeInfoAnalysis &shapeInfoAnalysis)
      : OpRewritePattern<triton::DotOp>(ctx), shapeAnalysis(shapeInfoAnalysis) {
  }

  LogicalResult matchAndRewrite(triton::DotOp dotOp,
                                PatternRewriter &rewriter) const override {
    Location loc = dotOp.getLoc();
    MLIRContext *ctx = dotOp.getContext();

    // Dot op computes standard (batch) GEMM.
    SmallVector<AffineMap> indexingMaps;
    TypedValue<RankedTensorType> res = dotOp.getD();
    uint32_t rank = res.getType().getRank();
    if (rank == 2) {
      indexingMaps.push_back(
          AffineMap::getMultiDimMapWithTargets(3, {0, 2}, ctx));
      indexingMaps.push_back(
          AffineMap::getMultiDimMapWithTargets(3, {2, 1}, ctx));
      indexingMaps.push_back(
          AffineMap::getMultiDimMapWithTargets(3, {0, 1}, ctx));
    } else if (rank == 3) {
      indexingMaps.push_back(
          AffineMap::getMultiDimMapWithTargets(4, {0, 1, 3}, ctx));
      indexingMaps.push_back(
          AffineMap::getMultiDimMapWithTargets(4, {0, 3, 2}, ctx));
      indexingMaps.push_back(
          AffineMap::getMultiDimMapWithTargets(4, {0, 1, 2}, ctx));
    }
    if (indexingMaps.size() == 0)
      return rewriter.notifyMatchFailure(dotOp, "unsupported indexing maps");

    TypedValue<RankedTensorType> lhs = dotOp.getA();
    TypedValue<RankedTensorType> rhs = dotOp.getB();
    TypedValue<RankedTensorType> acc = dotOp.getC();

    SmallVector<Attribute> flags;
    Value lhsBuf = getMemrefSource(rewriter, dotOp, lhs, shapeAnalysis);
    Value rhsBuf = getMemrefSource(rewriter, dotOp, rhs, shapeAnalysis);
    std::optional<Value> hoistedAcc =
        hoistAccumulationBuffer(rewriter, dotOp, acc, shapeAnalysis);
    Value accBuf = hoistedAcc
                       ? *hoistedAcc
                       : getMemrefSource(rewriter, dotOp, acc, shapeAnalysis);
    SmallVector<Value> inputs{lhsBuf, rhsBuf, accBuf};
    SmallVector<Value> outputs{nullptr};

    auto brgemmInfo = xsmm::utils::isMappableToBrgemm(rewriter, dotOp, inputs,
                                                      outputs, indexingMaps);
    if (failed(brgemmInfo))
      return rewriter.notifyMatchFailure(dotOp, "not mappable to XSMM");
    if (brgemmInfo->isVnni)
      return rewriter.notifyMatchFailure(dotOp, "VNNI support NYI");

    auto xsmmFuncs = xsmm::utils::buildBrgemmCalls(
        rewriter, dotOp, ValueRange{lhsBuf, rhsBuf, accBuf}, *brgemmInfo,
        flags);

    if (hoistedAcc) {
      // Hoisting already updated all uses correctly.
      // Only remove the original contraction.
      rewriter.eraseOp(dotOp);
    } else {
      // Load back the result to bring it back to tensor semantics.
      auto loadOp =
          rewriter.create<triton::cpu::LoadOp>(loc, res.getType(), accBuf);
      rewriter.replaceOp(dotOp, loadOp);
    }

    return success();
  }

private:
  ModuleTensorPtrShapeInfoAnalysis &shapeAnalysis;
};

struct ConvertTritonToXsmm
    : public triton::cpu::impl::ConvertTritonToXsmmBase<ConvertTritonToXsmm> {
  using ConvertTritonToXsmmBase::ConvertTritonToXsmmBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    ModuleTensorPtrShapeInfoAnalysis shapeInfoAnalysis(mod);

    RewritePatternSet patterns(context);
    patterns.add<DotToXsmm, DotReductionLoopToBrgemm>(context,
                                                      shapeInfoAnalysis);
    if (failed(mlir::applyPatternsAndFoldGreedily(mod, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace
