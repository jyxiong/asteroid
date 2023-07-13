#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "glm/glm.hpp"

#include "asteroid/renderer/ray.h"
#include "asteroid/renderer/camera.h"

namespace Asteroid
{

__device__ void GeneratePrimaryRay(const Camera& camera, const glm::uvec2 &uv, Ray &ray)
{
    glm::vec4 target = camera.GetInverseProjection() * glm::vec4(uv.x, uv.y, 1, 1);
    ray.Direction = glm::vec3(
            camera.GetInverseView() * glm::vec4(glm::normalize(glm::vec3(target) / target.w), 0));
    ray.Origin = camera.GetPosition();
}

__device__ glm::vec4 TraceRay(const Ray &ray)
{
    float radius = 0.5f;

    float a = glm::dot(ray.Direction, ray.Direction);
    float b = 2.0f * glm::dot(ray.Origin, ray.Direction);
    float c = glm::dot(ray.Origin, ray.Origin) - radius * radius;

    float discriminant = b * b - 4.0f * a * c;
    if (discriminant < 0.0f)
        return { 0, 0, 0, 1 };

    float closestT = (-b - glm::sqrt(discriminant)) / (2.0f * a);
    float t0 = (-b + glm::sqrt(discriminant)) / (2.0f * a); // Second hit distance (currently unused)

    glm::vec3 hitPoint = ray.Origin + ray.Direction * closestT;
    glm::vec3 normal = glm::normalize(hitPoint);

    glm::vec3 lightDir = glm::normalize(glm::vec3(-1, -1, -1));
    float lightIntensity = glm::max(glm::dot(normal, -lightDir), 0.0f);

    glm::vec3 sphereColor(1, 0, 1);
    sphereColor *= lightIntensity;
    return { sphereColor, 1.f };
}

__device__ glm::u8vec4 ConvertToRGBA(const glm::vec4 &color)
{
    return static_cast<glm::u8vec4>(color * 255.f);
}

}
