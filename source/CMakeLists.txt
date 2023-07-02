file(GLOB CORE_SOURCES asteroid/core/*.h asteroid/core/*.cpp)
file(GLOB EVENT_SOURCES asteroid/event/*.h asteroid/event/*.cpp)
file(GLOB IMGUI_SOURCES asteroid/imgui/*.h asteroid/imgui/*.cpp)
file(GLOB OPENGL_SOURCES platform/opengl/*.h platform/opengl/*.cpp)
file(GLOB WINDOWS_SOURCES platform/windows/*.h platform/windows/*.cpp)

add_library(asteroid STATIC
    ${CORE_SOURCES}
    ${EVENT_SOURCES}
    ${IMGUI_SOURCES}
    ${OPENGL_SOURCES}
    ${WINDOWS_SOURCES}
    )
target_include_directories(asteroid PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(asteroid PUBLIC spdlog glfw glad imgui)
target_precompile_headers(asteroid PUBLIC astpch.h)
target_compile_definitions(asteroid PUBLIC
    GLFW_INCLUDE_NONE
    AST_PLATFORM_WINDOWS
    AST_ENABLE_ASSERTS)