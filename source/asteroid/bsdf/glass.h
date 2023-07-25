#include <cuda_runtime.h>
#include <thrust/random.h>
#include "glm/glm.hpp"
#include "asteroid/renderer/scene_struct.h"

namespace Asteroid::Glass
{

__device__ glm::vec3 bsdf(
    const glm::vec3 &v, const glm::vec3 &l, const Intersection &its, const Material &mtl);

__device__ const glm::vec3 sampler(
    const glm::vec3 &v,
    const Intersection &its,
    const Material &mtl,
    thrust::default_random_engine &rng,
    float &pdf);

__device__ float pdf(const glm::vec3 &v, const glm::vec3 &l, const Intersection &intersect, const Material &mtl);

__device__ bool eval(
    const glm::vec3 &v,
    glm::vec3 &l,
    glm::vec3 &eval,
    const Intersection &intersect,
    const Material &mtl,
    thrust::default_random_engine &rng);

}