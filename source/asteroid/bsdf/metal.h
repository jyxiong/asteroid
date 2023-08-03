#include <cuda_runtime.h>
#include <thrust/random.h>
#include "asteroid/renderer/scene_struct.h"

namespace Asteroid::Metal
{

__device__ float3 bsdf(
    const float3 &v, const float3 &l, const Intersection &its, const Material &mtl);

__device__ const float3 sampler(
    const float3 &v,
    const Intersection &its,
    const Material &mtl,
    float &pdf);

__device__ float pdf(const float3 &v, const float3 &l, const Intersection &intersect, const Material &mtl);

__device__ bool eval(
    const float3 &v,
    float3 &l,
    float3 &eval,
    const Intersection &intersect,
    const Material &mtl);

}