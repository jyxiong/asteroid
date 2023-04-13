file(GLOB SOURCES CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)

add_executable(one_weekend ${SOURCES})
target_link_libraries(one_weekend PUBLIC glm stb)
