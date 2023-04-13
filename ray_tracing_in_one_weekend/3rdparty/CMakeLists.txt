cmake_minimum_required(VERSION 3.19)

project(ThirdParty)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CMakeDependentOption)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/module)

set(THIRD_PARTY_DIR "ThirdParty")

if (NOT TARGET glm)
    option(BUILD_STATIC_LIBS "" ON)
    option(BUILD_TESTING "" OFF)
    option(GLM_TEST_ENABLE "" OFF)
    add_subdirectory(glm)
    set_target_properties(glm_static PROPERTIES FOLDER ${THIRD_PARTY_DIR}/glm)
    set_target_properties(uninstall PROPERTIES FOLDER ${THIRD_PARTY_DIR}/glm)
endif ()

if (NOT TARGET stb)
    include(stb)
    set_target_properties(stb PROPERTIES FOLDER ${THIRD_PARTY_DIR}/stb)
endif ()
