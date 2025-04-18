add_triton_library(TritonToLinalg
    ConvertTritonToLinalg.cpp
    ConvertElementwise.cpp
    ConvertReduce.cpp
    ConvertTritonToMemRef.cpp

    DEPENDS
    TritonToLinalgPassIncGen
    MLIRDialectUtils

    LINK_LIBS PUBLIC
    TritonCPUIR
    MLIRIR
    MLIRSupport
)
