#include "cuda_runtime.h"
#include "glm/glm.hpp"

__device__ glm::u8vec4 ConvertToRGBA(const glm::vec4& color);

__device__ glm::u8vec4 PerPixel(glm::vec2 coord);
