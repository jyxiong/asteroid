#pragma once

#include "glm/glm.hpp"
#include "asteroid/renderer/camera.h"

namespace Asteroid
{
    void launch_cudaProcess(const Camera& camera, glm::u8vec4* g_odata, int width, int height);
}