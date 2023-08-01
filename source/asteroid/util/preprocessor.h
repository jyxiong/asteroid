#pragma once

#if defined(__CUDACC__) || defined(__CUDABE__)
#    define AST_DEVICE __device__
#    define AST_HOST_DEVICE __host__ __device__
#    define AST_INLINE __forceinline__
#    define CONST_STATIC_INIT( ... )
#else
#    define AST_DEVICE
#    define AST_HOST_DEVICE
#    define AST_INLINE inline
#    define CONST_STATIC_INIT( ... ) = __VA_ARGS__
#endif
