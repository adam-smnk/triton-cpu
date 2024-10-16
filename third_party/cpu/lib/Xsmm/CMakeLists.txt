add_triton_library(TritonCPUXsmm
    ConvertGemmToBrgemm.cpp
    ConvertVectorToXsmm.cpp
    VectorizeLinalg.cpp
    VnniUtils.cpp
    ValueUtils.cpp
    XsmmEnum.cpp
    XsmmUtils.cpp

    DEPENDS
    TritonCPUXsmmPassIncGen
    TritonCPUXsmmAttrDefIncGen
    xsmm

    LINK_LIBS PUBLIC
    TritonCPUIR
    MLIRIR
    MLIRPass
    MLIRLinalgDialect
    MLIRLinalgTransforms
    MLIRVectorDialect
    MLIRVectorTransforms
    MLIRMemRefDialect
    MLIRTensorDialect
    MLIRArithDialect
    MLIRMathDialect
    MLIRSCFDialect
    MLIRFuncDialect
    MLIRLLVMDialect
    MLIRInferTypeOpInterface
    MLIRLinalgUtils
    xsmm
)

target_include_directories(TritonCPUXsmm
  PUBLIC
    $<BUILD_INTERFACE:${XSMM_INCLUDE_DIRS}>
)
