#ifndef TRITONTOLINALG_CONVERSION_PASSES
#define TRITONTOLINALG_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTritonToLinalg : Pass<"triton-cpu-convert-triton-to-linalg"> {
    let summary = "Convert Triton ops to Linalg ops.";
    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::tensor::TensorDialect",
                             "mlir::linalg::LinalgDialect"];
}

#endif // TRITONTOLINALG_CONVERSION_PASSES
