#include "cpu/include/TritonToLinalg/TypeConverter.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinOps.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

mlir::triton::cpu::TritonTypeConverter::TritonTypeConverter(MLIRContext *ctx) {
  addConversion([](Type type) { return type; });
  addConversion([ctx](triton::PointerType ptrType) -> Type {
    auto tensorTy = dyn_cast<TensorType>(ptrType.getPointeeType());
    if (!tensorTy)
      return UnrankedMemRefType::get(ptrType.getPointeeType(),
                                     /*memorySpace=*/0);
    auto layout = StridedLayoutAttr::get(
        ctx, ShapedType::kDynamic,
        SmallVector<int64_t>(tensorTy.getRank(), ShapedType::kDynamic));
    return MemRefType::get(tensorTy.getShape(), tensorTy.getElementType(),
                           layout);
  });
  auto createUnrealizedCast = [&](OpBuilder &builder, Type resultType,
                                  ValueRange inputs, Location loc) -> Value {
    return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
        .getResult(0);
  };
  addSourceMaterialization(createUnrealizedCast);
  addTargetMaterialization(createUnrealizedCast);
}
