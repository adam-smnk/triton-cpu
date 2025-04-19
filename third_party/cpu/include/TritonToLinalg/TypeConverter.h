#ifndef TRITONTOLINALG_TYPE_CONVERTER_H
#define TRITONTOLINALG_TYPE_CONVERTER_H

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/MLIRContext.h"

namespace mlir {
namespace triton {
namespace cpu {

class TritonTypeConverter : public TypeConverter {
public:
  TritonTypeConverter(MLIRContext *ctx);
};

} // namespace cpu
} // namespace triton
} // namespace mlir

#endif // TRITONTOLINALG_TYPE_CONVERTER_H
