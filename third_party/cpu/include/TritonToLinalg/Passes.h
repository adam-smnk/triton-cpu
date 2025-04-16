#ifndef TRITONTOLINALG_CONVERSION_PASSES_H
#define TRITONTOLINALG_CONVERSION_PASSES_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Analysis/AxisInfo.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
namespace triton {
namespace cpu {

#define GEN_PASS_DECL
#include "cpu/include/TritonToLinalg/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "cpu/include/TritonToLinalg/Passes.h.inc"

// Collect a set of patterns to convert Triton Arith elementwise ops
// on tensors to Linalg ops.
void populateTritonElementwiseToLinalgPatterns(RewritePatternSet &patterns);

// Collect a set of patterns to convert Triton reduction ops to Linalg ops.
void populateTritonReduceToLinalgPatterns(RewritePatternSet &patterns);

} // namespace cpu
} // namespace triton
} // namespace mlir

#endif // TRITONTOLINALG_CONVERSION_PASSES_H
