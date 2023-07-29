include(embed_ptx)

embed_ptx(
    CONST
    HEADER otkHelloKernelPTX.h
    OUTPUT_TARGET otkHelloKernel
    FOLDER sandbox/optix
    PTX_LINK_LIBRARIES
        OptiX::OptiX
    SOURCES
        program.cu)

file(GLOB SOURCES *.h *.cpp)

add_executable(optix_renderer
    renderer.cu
    ${SOURCES})

target_link_libraries(optix_renderer PUBLIC
    asteroid
    otkHelloKernel)

set_target_properties(optix_renderer PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)