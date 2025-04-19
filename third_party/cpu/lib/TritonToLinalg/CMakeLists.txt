add_triton_library(TritonToLinalg
    ConvertTritonToLinalg.cpp
    ConvertElementwise.cpp
    ConvertReduce.cpp
    ConvertTritonToMemRef.cpp
    ConvertTritonToFunc.cpp
    TypeConverter.cpp

    DEPENDS
    TritonToLinalgPassIncGen
    MLIRDialectUtils

    LINK_LIBS PUBLIC
    TritonCPUIR
    MLIRIR
    MLIRSupport
)
