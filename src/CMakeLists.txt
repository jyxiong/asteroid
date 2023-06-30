file(GLOB CONTROL_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/control/*.h ${CMAKE_CURRENT_SOURCE_DIR}/control/*.cpp)
file(GLOB OPENGL_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/glwrapper/*.h ${CMAKE_CURRENT_SOURCE_DIR}/glwrapper/*.cpp)
file(GLOB RENDERER_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/renderer/*.h ${CMAKE_CURRENT_SOURCE_DIR}/renderer/*.cpp)
file(GLOB UTIL_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/util/*.h ${CMAKE_CURRENT_SOURCE_DIR}/util/*.cpp)

add_library(raytracer STATIC
        ${CONTROL_SOURCES}
        ${OPENGL_SOURCES}
        ${RENDERER_SOURCES}
        ${UTIL_SOURCES})
target_include_directories(raytracer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(raytracer PUBLIC glfw glad glm spdlog)

add_executable(main main.cpp)
target_link_libraries(main PUBLIC raytracer)
