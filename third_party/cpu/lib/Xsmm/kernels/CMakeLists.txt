find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_custom_command(
    OUTPUT xsmm-venv
    COMMAND ${Python3_EXECUTABLE} -m venv xsmm-venv
)
add_custom_command(
    OUTPUT xsmm-venv.stamp
    DEPENDS xsmm-venv requirements.txt
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt requirements.txt
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/xsmm-venv/bin/pip install -r requirements.txt --upgrade
)


add_custom_target(TritonCPUXsmmGenKernels
  DEPENDS
  ${CMAKE_CURRENT_BINARY_DIR}/XsmmKernels.h
)

add_custom_command(
  OUTPUT
  ${CMAKE_CURRENT_BINARY_DIR}/XsmmKernels.h

  DEPENDS
  xsmm
  xsmm-venv.stamp

  COMMAND
  ${CMAKE_CURRENT_BINARY_DIR}/xsmm-venv/bin/python ${LIBXSMMROOT}/scripts/libxsmm_staticlib.py -c ${CMAKE_CURRENT_SOURCE_DIR}/kernel-config.json -o ${CMAKE_CURRENT_BINARY_DIR}
)

file(GLOB XsmmKernelObjs "${CMAKE_CURRENT_BINARY_DIR}/xsmm_*.o")

add_library(TritonCPUXsmmKernels SHARED ${CMAKE_CURRENT_SOURCE_DIR}/Kernels.cpp)

target_link_libraries(TritonCPUXsmmKernels PRIVATE ${XsmmKernelObjs})
add_dependencies(TritonCPUXsmmKernels TritonCPUXsmmGenKernels)
target_include_directories(TritonCPUXsmmKernels
  PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}
)

# target_link_libraries(TritonCPUXsmmGenKernels PRIVATE xsmm)
# set_property(TARGET TritonCPUXsmmGenKernels PROPERTY CXX_STANDARD 11)

# target_include_directories(TritonCPUXsmmGenKernels
#   PUBLIC
#     $<BUILD_INTERFACE:${XSMM_INCLUDE_DIRS}>
# )
