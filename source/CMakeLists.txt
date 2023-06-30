file(GLOB CORE_SOURCES asteroid/core/*.h asteroid/core/*.cpp)
file(GLOB EVENT_SOURCES asteroid/event/*.h asteroid/event/*.cpp)
file(GLOB WINDOWS_SOURCES platform/windows/*.h platform/windows/*.cpp)

add_library(asteroid STATIC
    ${CORE_SOURCES}
    ${EVENT_SOURCES}
    ${WINDOWS_SOURCES}
    )
target_include_directories(asteroid PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(asteroid PUBLIC spdlog glfw glad)
