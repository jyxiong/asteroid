file(GLOB SOURCES *.h *.cpp *.cu)

add_executable(path_tracer ${SOURCES})

target_link_libraries(path_tracer PUBLIC asteroid)

set_target_properties(path_tracer PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)