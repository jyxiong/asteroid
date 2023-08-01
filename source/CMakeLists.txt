file(GLOB BASE_SOURCES asteroid/base/*.h asteroid/base/*.cpp asteroid/base/*.cu)
file(GLOB BSDF_SOURCES asteroid/bsdf/*.h)
file(GLOB EVENT_SOURCES asteroid/event/*.h asteroid/event/*.cpp)
file(GLOB IMGUI_SOURCES asteroid/imgui/*.h asteroid/imgui/*.cpp)
file(GLOB INPUT_SOURCES asteroid/input/*.h asteroid/input/*.cpp)
file(GLOB OPENGL_SOURCES asteroid/opengl/*.h asteroid/opengl/*.cpp)
file(GLOB RENDERER_SOURCES asteroid/renderer/*.h asteroid/renderer/*.cpp asteroid/renderer/*.cu)
file(GLOB UTIL_SOURCES asteroid/util/*.h asteroid/util/*.cpp)

add_library(asteroid STATIC
    ${BASE_SOURCES}
    ${BSDF_SOURCES}
    ${EVENT_SOURCES}
    ${IMGUI_SOURCES}
    ${INPUT_SOURCES}
    ${OPENGL_SOURCES}
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
    optix
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
