file(GLOB_RECURSE APP_SOURCES asteroid/app/*.h asteroid/app/*.cpp)
file(GLOB_RECURSE CUDA_SOURCES asteroid/cuda/*.h asteroid/cuda/*.cpp asteroid/cuda/*.inl)
file(GLOB_RECURSE KERNEL_SOURCES asteroid/kernel/*.h asteroid/kernel/*.cpp asteroid/kernel/*.inl)
file(GLOB RENDERER_SOURCES asteroid/renderer/*.h asteroid/renderer/*.cpp asteroid/renderer/*.cu)
file(GLOB UTIL_SOURCES asteroid/util/*.h asteroid/util/*.cpp)

add_library(asteroid STATIC
    ${APP_SOURCES}
    ${RENDERER_SOURCES}
    ${UTIL_SOURCES}
)

target_include_directories(asteroid SYSTEM PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

target_include_directories(asteroid PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(asteroid PUBLIC
    glad
    glfw
    glm
    imgui
    spdlog
    stb)

target_precompile_headers(asteroid PUBLIC astpch.h)

target_compile_definitions(asteroid PUBLIC
    GLFW_INCLUDE_NONE
    STB_IMAGE_IMPLEMENTATION
    AST_PLATFORM_WINDOWS
    AST_ENABLE_ASSERTS)

set_target_properties(asteroid PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)
