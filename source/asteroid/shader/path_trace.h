#pragma once

#include <cuda_runtime.h>
#include "glm/glm.hpp"

#include "asteroid/renderer/scene.h"
#include "asteroid/renderer/scene_struct.h"
#include "asteroid/shader/random.h"
#include "asteroid/shader/struct.h"
#include "asteroid/shader/ray_trace/ray_generation.h"

namespace Asteroid
{

__global__ void renderFrameKernel(const SceneView scene,
                          const Camera camera,
                          const RenderState state,
                          BufferView<glm::vec4> image)
{
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;

    auto& viewport = state.size;

    if (x >= viewport.x && y >= viewport.y)
        return;

    auto pixelColor = rayGeneration(scene, camera, state, {x, y});

    auto pixelIndex = y * viewport.x + x;
    auto oldColor = glm::vec3(image[pixelIndex]);
    auto newColor = glm::mix(oldColor, pixelColor, 1.f / float(state.frame + 1));
    // TODO: tone mapping
    // TODO: gamma correction

    image[pixelIndex] = glm::vec4(newColor, 1.f);
}

} // namespace Asteroid