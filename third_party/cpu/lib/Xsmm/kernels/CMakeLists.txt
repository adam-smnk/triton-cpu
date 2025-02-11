find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Hack to get python dependencies
# Somehow python execution in this context is isolated from user local modules.
# Thus, a separate virtual env is created to install dependencies for static lib script.
add_custom_command(
    OUTPUT xsmm-venv
    COMMAND ${Python3_EXECUTABLE} -m venv xsmm-venv
)
add_custom_command(
    OUTPUT xsmm-venv.stamp
    DEPENDS xsmm-venv requirements.txt
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt ${CMAKE_CURRENT_BINARY_DIR}/requirements.txt
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/xsmm-venv/bin/pip install -r ${CMAKE_CURRENT_BINARY_DIR}/requirements.txt --upgrade
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
  xsmm-generator

  # Hack to make static lib script work
  # TODO: Generalize staticlib script to not rely on relative paths for binaries
  COMMAND ${CMAKE_COMMAND} -E copy ${LIBXSMMROOT}/scripts/libxsmm_staticlib.py ${CMAKE_CURRENT_BINARY_DIR}/bin/libxsmm_staticlib.py
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/kernel-config.json ${CMAKE_CURRENT_BINARY_DIR}/kernel-config.json
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/bin
  COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_BINARY_DIR}/third_party/cpu/xsmm-generator ${CMAKE_CURRENT_BINARY_DIR}/bin/libxsmm_binaryexport_generator
  # Generate static kernel objects
  COMMAND ${CMAKE_CURRENT_BINARY_DIR}/xsmm-venv/bin/python ${CMAKE_CURRENT_BINARY_DIR}/bin/libxsmm_staticlib.py -c ${CMAKE_CURRENT_BINARY_DIR}/kernel-config.json -o ${CMAKE_CURRENT_BINARY_DIR}
)

file(GLOB XSMM_KERNEL_OBJS CONFIGURE_DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/xsmm_*.o")
message(XSMM_KERNEL_OBJS="${XSMM_KERNEL_OBJS}")

add_library(TritonCPUXsmmKernels SHARED ${CMAKE_CURRENT_SOURCE_DIR}/Kernels.cpp)

target_link_libraries(TritonCPUXsmmKernels PRIVATE ${XSMM_KERNEL_OBJS})
add_dependencies(TritonCPUXsmmKernels TritonCPUXsmmGenKernels)
target_include_directories(TritonCPUXsmmKernels
  PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}/XsmmKernels
)
