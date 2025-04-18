#ifndef TRITONTOLINALG_CONVERSION_PASSES
#define TRITONTOLINALG_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTritonToLinalg : Pass<"triton-cpu-convert-triton-to-linalg"> {
    let summary = "Convert Triton ops to Linalg ops.";
    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::tensor::TensorDialect",
                             "mlir::linalg::LinalgDialect"];
}

def ConvertTritonToMemRef : Pass<"triton-cpu-convert-triton-to-memref"> {
    let summary = "Convert Triton ops to MemRef ops.";
    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::affine::AffineDialect",
                             "mlir::tensor::TensorDialect",
                             "mlir::memref::MemRefDialect",
                             "mlir::bufferization::BufferizationDialect",
                             "mlir::linalg::LinalgDialect"];
}

#endif // TRITONTOLINALG_CONVERSION_PASSES
