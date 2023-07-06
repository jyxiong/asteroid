file(GLOB SOURCES *.h *.cpp *.cu)

add_executable(cuda_opengl ${SOURCES})

target_link_libraries(cuda_opengl PUBLIC asteroid)
