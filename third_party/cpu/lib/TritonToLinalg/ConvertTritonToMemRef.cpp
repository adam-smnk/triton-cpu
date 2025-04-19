#include "cpu/include/TritonToLinalg/Passes.h"
#include "cpu/include/TritonToLinalg/TypeConverter.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SetVector.h"

#include "llvm/Support/Debug.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace cpu {
#define GEN_PASS_DEF_CONVERTTRITONTOMEMREF
#include "cpu/include/TritonToLinalg/Passes.h.inc"
} // namespace cpu
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

Operation *getArgDefinitingOp(BlockArgument arg) {
  if (!arg)
    return nullptr;

  Operation *parentOp = arg.getOwner()->getParent()->getParentOp();
  assert(parentOp && "invalid block arg");
  if (dyn_cast<triton::FuncOp>(parentOp))
    return parentOp;
  if (dyn_cast<func::FuncOp>(parentOp))
    return parentOp;
  if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
    Value val = forOp.getTiedLoopInit(arg)->get();
    if (auto parentArg = dyn_cast<BlockArgument>(val))
      return getArgDefinitingOp(parentArg);
    return val.getDefiningOp();
  }
  return nullptr;
}

// Get other ptr aliases when possible.
// Bail out on unknown source.
Value getAliasingPtr(Operation *op) {
  if (!op)
    return nullptr;

  if (auto makeTensorPtr = dyn_cast<triton::MakeTensorPtrOp>(op))
    return makeTensorPtr.getBase();
  else if (auto advance = dyn_cast<triton::AdvanceOp>(op))
    return advance.getPtr();
  else if (auto addptr = dyn_cast<triton::AddPtrOp>(op))
    return addptr.getPtr();
  else
    return nullptr;
}

bool isReadOnly(Value rootPtr) {
  assert(rootPtr && "Invalid root ptr");
  SmallVector<Value> aliases = {rootPtr};
  llvm::SmallSetVector<Operation *, 16> visitedOps;
  while (!aliases.empty()) {
    Value val = aliases.pop_back_val();
    Operation *defOp = val.getDefiningOp();
    if (!defOp)
      defOp = getArgDefinitingOp(dyn_cast<BlockArgument>(val));

    // Bail out on unknown source.
    if (!defOp)
      return false;
    // Function argument is the terminal definition - look no further.
    if (!isa<FunctionOpInterface>(defOp)) {
      Value alias = getAliasingPtr(defOp);
      if (!alias)
        return false;
      aliases.push_back(alias);
    }

    for (Operation *opUser : val.getUsers()) {
      if (!visitedOps.insert(opUser))
        continue;
      // Function calls have unknown side effects - assume write.
      // Store op writes memory.
      // Splat is not handled atm.
      if (isa<CallOpInterface, triton::StoreOp, triton::SplatOp>(opUser))
        return false;
      if (auto forOp = dyn_cast<scf::ForOp>(opUser)) {
        // Treat iter_arg as an alias and follow into the loop body.
        Value loopval;
        for (OpOperand &arg : forOp.getInitsMutable()) {
          if (arg.get() == val) {
            loopval = forOp.getTiedLoopRegionIterArg(&arg);
            break;
          }
        }
        assert(loopval && "failed scf");
        aliases.push_back(loopval);
        continue;
      }
      if (unsigned numResults = opUser->getNumResults()) {
        // Lost dependency chain with multiple results - bail out.
        if (numResults > 1)
          return false;
        Value res = opUser->getResult(0);
        if (isa<triton::PointerType>(res.getType()))
          aliases.push_back(res);
      }
    }
  }

  return true;
}

class ConvertLoadOp : public OpConversionPattern<triton::LoadOp> {
public:
  using OpConversionPattern<triton::LoadOp>::OpConversionPattern;

  ConvertLoadOp(const TypeConverter &typeConverter, MLIRContext *context,
                int benefit)
      : OpConversionPattern<triton::LoadOp>(typeConverter, context, benefit) {}

  LogicalResult
  matchAndRewrite(triton::LoadOp loadOp, triton::LoadOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = loadOp.getLoc();

    auto ptrType = dyn_cast<triton::PointerType>(loadOp.getPtr().getType());
    if (!ptrType)
      return rewriter.notifyMatchFailure(loadOp, "NYI load tensor of pointers");

    // Load a scalar value.
    Type pointeeTy = ptrType.getPointeeType();
    if (!isa<TensorType>(pointeeTy)) {
      auto scalarCast = rewriter.create<memref::CastOp>(
          loc, MemRefType::get({}, pointeeTy), adaptor.getPtr());
      rewriter.replaceOpWithNewOp<memref::LoadOp>(loadOp, pointeeTy, scalarCast,
                                                  /*indices=*/ValueRange{});
      return success();
    }

    bool createCopy = loadOp.getIsVolatile() || !isReadOnly(loadOp.getPtr());

    // Load a block pointer.
    Value loadedTensor = rewriter.create<bufferization::ToTensorOp>(
        loc, adaptor.getPtr(), /*restrict=*/true);

    if (createCopy) {
      auto empty = rewriter.create<tensor::EmptyOp>(
          loc, loadedTensor.getType(), /*dynamicSizes=*/ValueRange{});
      loadedTensor = rewriter
                         .create<linalg::CopyOp>(loc, ValueRange{loadedTensor},
                                                 ValueRange{empty})
                         .getResultTensors()[0];
    }

    rewriter.replaceOp(loadOp, loadedTensor);

    return success();
  }
};

class ConvertStoreOp : public OpConversionPattern<triton::StoreOp> {
public:
  using OpConversionPattern<triton::StoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::StoreOp storeOp, triton::StoreOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = storeOp.getLoc();

    auto ptrType = dyn_cast<triton::PointerType>(storeOp.getPtr().getType());
    if (!ptrType)
      return rewriter.notifyMatchFailure(storeOp,
                                         "NYI store tensor of pointers");

    // Store a scalar value.
    Type pointeeTy = ptrType.getPointeeType();
    if (!isa<TensorType>(pointeeTy)) {
      auto scalarCast = rewriter.create<memref::CastOp>(
          loc, MemRefType::get({}, pointeeTy), adaptor.getPtr());
      rewriter.replaceOpWithNewOp<memref::StoreOp>(storeOp, adaptor.getValue(),
                                                   scalarCast);
      return success();
    }

    // Store a block pointer.
    if (!ptrType || !isa<TensorType>(ptrType.getPointeeType()))
      return rewriter.notifyMatchFailure(storeOp,
                                         "NYI non tensor pointer stores");

    auto bufWrite = rewriter.create<bufferization::MaterializeInDestinationOp>(
        loc, adaptor.getValue(), adaptor.getPtr());
    bufWrite.setWritable(true);

    rewriter.replaceOp(storeOp, bufWrite);

    return success();
  }
};

class ConvertMakeTensorPtrOp
    : public OpConversionPattern<triton::MakeTensorPtrOp> {
public:
  using OpConversionPattern<triton::MakeTensorPtrOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::MakeTensorPtrOp makeTensorPtrOp,
                  triton::MakeTensorPtrOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = makeTensorPtrOp.getLoc();

    ArrayRef<int32_t> order = makeTensorPtrOp.getOrder();
    auto orderSize = order.size();
    if (orderSize > 1) {
      for (auto [first, second] : llvm::zip_equal(
               order.slice(0, orderSize - 1), order.slice(1, orderSize - 1))) {
        if (first != second + 1)
          return rewriter.notifyMatchFailure(
              makeTensorPtrOp, "Currently only support default row-major order "
                               "on block pointers");
      }
    }

    IndexType indexTy = rewriter.getIndexType();
    auto castToIndex = [&](ValueRange vals) -> SmallVector<OpFoldResult> {
      SmallVector<OpFoldResult> indexVals;
      for (Value val : vals)
        indexVals.push_back(
            rewriter.create<arith::IndexCastOp>(loc, indexTy, val).getOut());
      return indexVals;
    };

    OpFoldResult offset(rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> sizes = castToIndex(adaptor.getShape());
    SmallVector<OpFoldResult> strides = castToIndex(adaptor.getStrides());
    auto baseBuffer = rewriter.create<memref::ReinterpretCastOp>(
        loc, adaptor.getBase(), offset, sizes, strides);

    auto blockTy = dyn_cast<TensorType>(
        makeTensorPtrOp.getResult().getType().getPointeeType());
    assert(blockTy && "Not a block pointer to a tensor");

    SmallVector<OpFoldResult> blockOffsets = castToIndex(adaptor.getOffsets());
    SmallVector<OpFoldResult> blockSizes;
    for (auto size : blockTy.getShape())
      blockSizes.push_back(rewriter.getIndexAttr(size));
    SmallVector<OpFoldResult> blockStrides(
        blockTy.getRank(), OpFoldResult(rewriter.getIndexAttr(1)));
    auto targetBlock = rewriter.create<memref::SubViewOp>(
        loc, baseBuffer, blockOffsets, blockSizes, blockStrides);

    rewriter.replaceOp(makeTensorPtrOp, targetBlock);

    return success();
  }
};

class ConvertAdvanceOp : public OpConversionPattern<triton::AdvanceOp> {
public:
  using OpConversionPattern<triton::AdvanceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::AdvanceOp advanceOp,
                  triton::AdvanceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = advanceOp.getLoc();

    auto baseMetadata = rewriter.create<memref::ExtractStridedMetadataOp>(
        loc, adaptor.getPtr());

    IndexType indexTy = rewriter.getIndexType();
    SmallVector<OpFoldResult> offsets;
    for (Value offset : adaptor.getOffsets())
      offsets.push_back(
          rewriter.create<arith::IndexCastOp>(loc, indexTy, offset).getOut());

    SmallVector<OpFoldResult> strides =
        getAsOpFoldResult(baseMetadata.getStrides());
    auto &&[expr, values] = computeLinearIndex(
        OpFoldResult(baseMetadata.getOffset()), strides, offsets);
    OpFoldResult newOffset =
        affine::makeComposedFoldedAffineApply(rewriter, loc, expr, values);

    auto blockTy =
        dyn_cast<TensorType>(advanceOp.getResult().getType().getPointeeType());
    assert(blockTy && "Not a block pointer to a tensor");
    SmallVector<OpFoldResult> sizes;
    for (auto size : blockTy.getShape())
      sizes.push_back(rewriter.getIndexAttr(size));

    auto newBuffer = rewriter.create<memref::ReinterpretCastOp>(
        loc, baseMetadata.getBaseBuffer(), newOffset, sizes, strides);

    rewriter.replaceOp(advanceOp, newBuffer);

    return success();
  }
};

struct ConvertTritonToMemRef
    : public triton::cpu::impl::ConvertTritonToMemRefBase<
          ConvertTritonToMemRef> {
  using ConvertTritonToMemRefBase::ConvertTritonToMemRefBase;

  void runOnOperation() override {
    auto *ctx = &getContext();

    TritonTypeConverter converter(ctx);

    ConversionTarget target(*ctx);
    target.addLegalDialect<memref::MemRefDialect, tensor::TensorDialect,
                           linalg::LinalgDialect, arith::ArithDialect,
                           affine::AffineDialect, func::FuncDialect,
                           bufferization::BufferizationDialect>();
    // config illegal ops

    RewritePatternSet patterns(ctx);
    patterns.add<ConvertLoadOp>(converter, ctx, /*benefit=*/10);
    patterns.add<ConvertStoreOp, ConvertMakeTensorPtrOp, ConvertAdvanceOp>(
        converter, ctx);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();

    RewritePatternSet scfPatterns(ctx);
    ConversionTarget scfTarget(*ctx);
    scfTarget.addLegalDialect<scf::SCFDialect>();
    scf::populateSCFStructuralTypeConversionsAndLegality(converter, scfPatterns,
                                                         scfTarget);
    if (failed(applyPartialConversion(getOperation(), scfTarget,
                                      std::move(scfPatterns))))
      return signalPassFailure();
  }
};

} // namespace
