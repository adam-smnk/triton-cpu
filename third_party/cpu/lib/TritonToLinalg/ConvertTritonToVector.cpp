#include "cpu/include/TritonToLinalg/Passes.h"
#include "cpu/include/TritonToLinalg/TypeConverter.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
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
#define GEN_PASS_DEF_CONVERTTRITONTOVECTOR
#include "cpu/include/TritonToLinalg/Passes.h.inc"
} // namespace cpu
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

static bool hasRankedTensorOperands(Operation *op) {
  return llvm::all_of(op->getOperandTypes(), llvm::IsaPred<RankedTensorType>);
}

static bool isElementwiseMappableOpOnRankedTensors(Operation *op) {
  if (!OpTrait::hasElementwiseMappableTraits(op))
    return false;

  return hasRankedTensorOperands(op);
}

struct ConvertElementwise : public ConversionPattern {
  ConvertElementwise(const TypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(typeConverter, MatchAnyOpTypeTag(), /*benefit=*/1,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    if (!isElementwiseMappableOpOnRankedTensors(op))
      return rewriter.notifyMatchFailure(
          op, "requires elementwise op on ranked tensors");

    Operation *newOp = rewriter.clone(*op);
    newOp->setOperands(operands);
    for (auto res : newOp->getResults())
      res.setType(typeConverter->convertType(res.getType()));

    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

struct ConvertConstant : public OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp constantOp,
                  arith::ConstantOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (!hasRankedTensorOperands(constantOp))
      return rewriter.notifyMatchFailure(
          constantOp, "requires constant op on ranked tensors");

    auto denseAttr = dyn_cast<DenseElementsAttr>(constantOp.getValueAttr());
    if (!denseAttr)
      return rewriter.notifyMatchFailure(
          constantOp, "expectes constant op with dense attr");

    auto newResType = dyn_cast<VectorType>(
        typeConverter->convertType(constantOp.getResult().getType()));
    assert(newResType && "expected vector type after conversion");
    SmallVector<Attribute> values(denseAttr.getValues<Attribute>());
    auto newValueAttr = DenseElementsAttr::get(newResType, values);

    rewriter.replaceOpWithNewOp<arith::ConstantOp>(constantOp, newResType,
                                                   newValueAttr);
    return success();
  }
};

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

    auto tensorTy = dyn_cast<TensorType>(ptrType.getPointeeType());
    if (!tensorTy)
      return rewriter.notifyMatchFailure(loadOp, "expects pointer to a tensor");

    auto zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    SmallVector<Value> indices(tensorTy.getRank(), zeroIdx);
    SmallVector<bool> inBounds(tensorTy.getRank(), true);
    for (auto dim : loadOp.getBoundaryCheck())
      inBounds[dim] = false;

    // By default, use zero padding.
    auto elementType = tensorTy.getElementType();
    TypedAttr attr = rewriter.getZeroAttr(elementType);
    Value padding = rewriter.create<arith::ConstantOp>(loc, attr);
    if (auto paddingOpt = loadOp.getPadding()) {
      // Float NaN padding case
      if (*paddingOpt == triton::PaddingOption::PAD_NAN) {
        assert(!elementType.isIntOrIndex());
        auto apNaN = llvm::APFloat::getNaN(
            cast<FloatAttr>(attr).getValue().getSemantics());
        attr = rewriter.getFloatAttr(elementType, apNaN);
        padding = rewriter.create<arith::ConstantOp>(loc, attr);
      }
    }

    auto newResType = dyn_cast<VectorType>(
        typeConverter->convertType(loadOp.getResult().getType()));
    assert(newResType && "expected vector type after conversion");

    rewriter.replaceOpWithNewOp<vector::TransferReadOp>(
        loadOp, newResType, adaptor.getPtr(), indices, padding, inBounds);
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

    auto tensorTy = dyn_cast<TensorType>(ptrType.getPointeeType());
    if (!tensorTy)
      return rewriter.notifyMatchFailure(storeOp,
                                         "expects pointer to a tensor");

    auto zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    SmallVector<Value> indices(tensorTy.getRank(), zeroIdx);
    SmallVector<bool> inBounds(tensorTy.getRank(), true);
    for (auto dim : storeOp.getBoundaryCheck())
      inBounds[dim] = false;

    rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(
        storeOp, adaptor.getValue(), adaptor.getPtr(), indices, inBounds);
    return success();
  }
};

struct ConvertDotOp : public OpConversionPattern<triton::DotOp> {
  using OpConversionPattern<triton::DotOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::DotOp dotOp, triton::DotOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto *ctx = dotOp.getContext();

    AffineExpr d0, d1, d2;
    bindDims(ctx, d0, d1, d2);
    SmallVector<vector::IteratorType> iterTypes = {
        vector::IteratorType::parallel, vector::IteratorType::parallel,
        vector::IteratorType::reduction};

    using MapList = ArrayRef<ArrayRef<AffineExpr>>;
    rewriter.replaceOpWithNewOp<vector::ContractionOp>(
        dotOp, adaptor.getA(), adaptor.getB(), adaptor.getC(),
        MapList{{d0, d2}, {d2, d1}, {d0, d1}}, iterTypes);
    return success();
  }
};

struct ConvertTritonToVector
    : public triton::cpu::impl::ConvertTritonToVectorBase<
          ConvertTritonToVector> {
  using ConvertTritonToVectorBase::ConvertTritonToVectorBase;

  void runOnOperation() override {
    auto *ctx = &getContext();

    TritonTypeConverter converter(ctx);
    converter.addConversion([](TensorType tensorTy) {
      return VectorType::get(tensorTy.getShape(), tensorTy.getElementType());
    });

    ConversionTarget target(*ctx);
    target.addLegalDialect<memref::MemRefDialect, affine::AffineDialect,
                           func::FuncDialect, vector::VectorDialect>();
    auto isLegal = [](Operation *op) {
      if (auto constantOp = dyn_cast<arith::ConstantOp>(op))
        return !isa<RankedTensorType>(constantOp.getResult().getType());
      return !hasRankedTensorOperands(op);
    };
    target.addDynamicallyLegalDialect<arith::ArithDialect>(isLegal);
    target.addDynamicallyLegalDialect<math::MathDialect>(isLegal);

    RewritePatternSet patterns(ctx);
    patterns.add<ConvertElementwise, ConvertConstant, ConvertLoadOp,
                 ConvertStoreOp, ConvertDotOp>(converter, ctx);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();

    // Convert types at boundaries like functions and loops.
    RewritePatternSet boundaryPatterns(ctx);
    ConversionTarget boundaryTarget(*ctx);
    boundaryTarget.addLegalDialect<scf::SCFDialect>();
    boundaryTarget.addDynamicallyLegalOp<triton::FuncOp>([&](auto op) {
      return converter.isSignatureLegal(
          cast<FunctionType>(cast<FunctionOpInterface>(op).getFunctionType()));
    });
    boundaryTarget.addDynamicallyLegalOp<triton::CallOp>(
        [&](triton::CallOp op) {
          return converter.isLegal(op.getResultTypes()) &&
                 converter.isLegal(op.getOperandTypes());
        });

    scf::populateSCFStructuralTypeConversionsAndLegality(
        converter, boundaryPatterns, boundaryTarget);
    populateFunctionOpInterfaceTypeConversionPattern<triton::FuncOp>(
        boundaryPatterns, converter);
    populateCallOpTypeConversionPattern(boundaryPatterns, converter);
    if (failed(applyPartialConversion(getOperation(), boundaryTarget,
                                      std::move(boundaryPatterns))))
      return signalPassFailure();
  }
};

} // namespace
