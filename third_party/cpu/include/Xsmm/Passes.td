#ifndef TRITONCPU_XSMM_PASSES
#define TRITONCPU_XSMM_PASSES

include "mlir/Pass/PassBase.td"

def ConvertVectorToXsmm : Pass<"triton-cpu-convert-vector-to-xsmm", "mlir::ModuleOp"> {
 let summary = "Convert vector to xsmm";
 let description = [{
   Convert vector operations to XSMM operations.
 }];
 let dependentDialects = ["func::FuncDialect",
                          "memref::MemRefDialect",
                          "vector::VectorDialect",
			                    "LLVM::LLVMDialect"];
}

def ConvertGemmToBrgemm : Pass<"triton-cpu-convert-gemm-to-brgemm", "mlir::ModuleOp"> {
 let summary = "Convert GEMM to BRGEMM operation";
 let description = [{
   Convert GEMM operations to BRGEMM operations.
 }];
 let dependentDialects = ["linalg::LinalgDialect",
                          "func::FuncDialect",
                          "tensor::TensorDialect",
                          "mlir::triton::TritonDialect"];
}

#endif
