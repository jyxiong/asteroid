#pragma once

#include <string>
#include <vector>
#include "glm/glm.hpp"

namespace Asteroid {

struct RenderState {
    unsigned int currentIteration{0};
    unsigned int traceDepth{5};
};

struct Camera
{
    glm::vec3 position{ 0, 0, 6 };
    glm::vec3 direction{ 0, 0, -1 };
    glm::vec3 up{ 0, 1, 0 };

    float verticalFov{45.f};
    float focalDistance{0.f};
    glm::uvec2 viewport{1, 1};

    glm::vec3 right;
    float tanHalfFov;
    float aspectRatio;

    glm::vec2 lastMousePosition{0.0f, 0.0f};
};

struct Material {
    glm::vec3 albedo{1.0f};
    float roughness = 1.0f;
    float metallic = 0.0f;
    float emittance = 0.0f;

};

struct Sphere {
    float radius = 0.5f;

    glm::vec3 position{0.0f};

    int materialIndex = 0;
};

struct Ray {
    glm::vec3 origin;

    glm::vec3 direction;
};

struct PathSegment {
    Ray ray;
    glm::vec3 color; // accumulated light
    glm::vec3 throughput;
    int pixelIndex;
    unsigned int remainingBounces;
};

struct Intersection {
    float t;
    glm::vec3 normal;
    glm::vec3 position;
    int materialIndex;
};

} // namespace Asteroid
