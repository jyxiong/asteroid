#pragma once

#include <stdexcept>
#include <sstream>

#ifdef AST_ENABLE_ASSERTS

#define AST_ASSERT(x, ...) { if(!(x)) { AST_ERROR("Assertion Failed: {0}", __VA_ARGS__); __debugbreak(); } }
#define AST_CORE_ASSERT(x, ...) { if(!(x)) { AST_CORE_ERROR("Assertion Failed: {0}", __VA_ARGS__); __debugbreak(); } }

#else

#define AST_ASSERT(x, ...)
#define AST_CORE_ASSERT(x, ...)

#endif

#define BIT(x) (1 << x)

#define CUDA_CHECK(call)                                        \
{                                                               \
    cudaError_t rc = call;                                      \
    if (rc != cudaSuccess) {                                    \
        std::stringstream txt;                                  \
        cudaError_t err =  rc; /*cudaGetLastError();*/          \
        txt << "CUDA Error " << cudaGetErrorName(err)           \
            << " (" << cudaGetErrorString(err) << ")";          \
        throw std::runtime_error(txt.str());                    \
    }                                                           \
}

#define CUDA_SYNC_CHECK()                                       \
{                                                               \
    cudaDeviceSynchronize();                                    \
    cudaError_t error = cudaGetLastError();                     \
    if( error != cudaSuccess )                                  \
    {                                                           \
        fprintf( stderr, "error (%s: line %d): %s\n", __FILE__, __LINE__, cudaGetErrorString( error ) ); \
        exit( 2 );                                              \
    }                                                           \
}