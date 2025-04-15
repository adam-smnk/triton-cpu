add_triton_library(TritonToLinalg
    ConvertTritonToLinalg.cpp

    DEPENDS
    TritonToLinalgPassIncGen
    MLIRDialectUtils

    LINK_LIBS PUBLIC
    TritonCPUIR
    MLIRIR
    MLIRSupport
)
