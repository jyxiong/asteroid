file(GLOB SOURCES *.h *.cpp)

add_executable(path_tracer ${SOURCES})

target_link_libraries(path_tracer PUBLIC asteroid)
