if (TARGET optix)
    return()
endif()

file(GLOB OPTIX_SDK_DIR "$ENV{ProgramData}/NVIDIA Corporation/OptiX SDK 7.7.*")
find_path(OptiX_ROOT_DIR NAMES include/optix.h PATHS ${OptiX_INSTALL_DIR} ${OPTIX_SDK_DIR} REQUIRED)

file(READ "${OptiX_ROOT_DIR}/include/optix.h" header)
string(REGEX REPLACE "^.*OPTIX_VERSION ([0-9]+)([0-9][0-9])([0-9][0-9])[^0-9].*$" "\\1.\\2.\\3" OPTIX_VERSION ${header})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OptiX
    FOUND_VAR OptiX_FOUND
    VERSION_VAR OPTIX_VERSION
    REQUIRED_VARS
    OptiX_ROOT_DIR
    REASON_FAILURE_MESSAGE
    "OptiX installation not found. Please use CMAKE_PREFIX_PATH or OptiX_INSTALL_DIR to locate 'include/optix.h'."
)

set(OptiX_INCLUDE_DIR ${OptiX_ROOT_DIR}/include)

add_library(optix INTERFACE IMPORTED)
target_include_directories(optix INTERFACE ${OptiX_INCLUDE_DIR})