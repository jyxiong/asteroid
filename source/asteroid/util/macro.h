#pragma once

#include <stdexcept>
#include <sstream>
#include "asteroid/util/log.h"

#ifdef AST_ENABLE_ASSERTS

#define AST_ASSERT(x, ...) { if(!(x)) { AST_ERROR("Assertion Failed: {0}", __VA_ARGS__); __debugbreak(); } }
#define AST_CORE_ASSERT(x, ...) { if(!(x)) { AST_CORE_ERROR("Assertion Failed: {0}", __VA_ARGS__); __debugbreak(); } }

#else

#define AST_ASSERT(x, ...)
#define AST_CORE_ASSERT(x, ...)

#endif

#define BIT(x) (1 << x)

#define AST_CUDA_CHECK(call)                                                                             \
    {                                                                                                \
        cudaError_t rc = call;                                                                       \
        if (rc != cudaSuccess)                                                                       \
        {                                                                                            \
            std::stringstream txt;                                                                   \
            cudaError_t err = rc; /*cudaGetLastError();*/                                            \
            txt << "CUDA Error " << cudaGetErrorName(err) << " (" << cudaGetErrorString(err) << ")"; \
            AST_CORE_CRITICAL(txt.str());                                                            \
        }                                                                                            \
    }

#define AST_CUDA_SYNC_CHECK()                                                                             \
    {                                                                                                 \
        cudaDeviceSynchronize();                                                                      \
        cudaError_t error = cudaGetLastError();                                                       \
        if (error != cudaSuccess)                                                                     \
        {                                                                                             \
            char buf[1024];                                                                           \
            sprintf(buf, "error (%s: line %d): %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            AST_CORE_CRITICAL(std::string(buf));                                                      \
        }                                                                                             \
    }

#define AST_CUDA_CHECK_SUCCESS(x)                                                                                            \
    do                                                                                                                   \
    {                                                                                                                    \
        cudaError_t result = x;                                                                                          \
        if (result != cudaSuccess)                                                                                       \
        {                                                                                                                \
            AST_CORE_CRITICAL("CUDA Error: " #x " failed with error " + std::string(cudaGetErrorString(result))); \
            return 0;                                                                                                    \
        }                                                                                                                \
    } while (0)

#define AST_OPTIX_CHECK( call )                                             \
  {                                                                     \
    OptixResult res = call;                                             \
    if( res != OPTIX_SUCCESS )                                          \
      {                                                                 \
        fprintf( stderr, "Optix call (%s) failed with code %d (line %d)\n", #call, res, __LINE__ ); \
        exit( 2 );                                                      \
      }                                                                 \
  }
