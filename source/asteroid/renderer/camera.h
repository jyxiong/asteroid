#pragma once

#include "glm/glm.hpp"
#include "asteroid/renderer/ray.h"

namespace Asteroid {

struct Camera {
    glm::vec3 position{0, 0, 6};
    glm::vec3 direction{0, 0, -1};
    glm::vec3 up{0, 1, 0};

    float verticalFov{45.f};
    float focalDistance{0.f};
    glm::uvec2 viewport{1, 1};

    glm::vec3 right;
    float tanHalfFov;
    float aspectRatio;

    glm::vec2 lastMousePosition{0.0f, 0.0f};
};

}
