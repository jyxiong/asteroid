#pragma once

#include <vector>
#include <string>
#include "glm/glm.hpp"

namespace Asteroid
{

struct RenderState {
    unsigned int iterations;
    unsigned int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

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

struct Material {
    glm::vec3 Albedo{ 1.0f };
    float Roughness = 1.0f;
    float Metallic = 0.0f;
};

struct Sphere {
    float Radius = 0.5f;

    glm::vec3 Position{0.0f};

    int MaterialId = 0;

};

struct Ray
{
    glm::vec3 Origin;

    glm::vec3 Direction;
};

struct PathSegment {
    Ray ray;
    glm::vec3 color; // accumulated light
    glm::vec3 throughput;
    int pixelIndex;
    unsigned int remainingBounces;
};

struct Intersection
{
    float t;
    glm::vec3 normal;
    glm::vec3 position;
    int materialId;
};

}
