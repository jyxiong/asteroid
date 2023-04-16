file(GLOB SOURCES CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)

add_executable(test_cuda ${SOURCES})
target_link_libraries(test_cuda PUBLIC glm stb)
