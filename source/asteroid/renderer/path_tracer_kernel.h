#pragma once

#include <cuda_runtime.h>
#include "glm/glm.hpp"

#include "asteroid/renderer/ray.h"
#include "asteroid/renderer/camera.h"
#include "asteroid/renderer/scene.h"

namespace Asteroid {

    __device__ glm::vec4 TraceRay(const SceneView &scene, const Ray &ray) {

        if (scene.deviceSpheres.size() == 0)
            return {1, 1, 1, 1};

        const Sphere *closestSphere = nullptr;
        float hitDistance = std::numeric_limits<float>::max();

        for (size_t i = 0; i < scene.deviceSpheres.size(); ++i) {
            auto sphere = scene.deviceSpheres[i];

            return {sphere.Radius, sphere.Radius,sphere.Radius,1};

            glm::vec3 origin = ray.Origin - sphere.Position;

            float a = glm::dot(ray.Direction, ray.Direction);
            float b = 2.0f * glm::dot(origin, ray.Direction);
            float c = glm::dot(origin, origin) - sphere.Radius * sphere.Radius;

            float discriminant = b * b - 4.0f * a * c;
            if (discriminant < 0.0f)
                continue;

            float closestT = (-b - glm::sqrt(discriminant)) / (2.0f * a);
            if (closestT < hitDistance) {
                hitDistance = closestT;
                closestSphere = &sphere;
            }
        }

        if (closestSphere == nullptr)
            return {1.0f, 0.0f, 0.0f, 1.0f};

        glm::vec3 origin = ray.Origin - closestSphere->Position;
        glm::vec3 hitPoint = ray(hitDistance);
        glm::vec3 normal = glm::normalize(hitPoint);

        glm::vec3 lightDir = glm::normalize(glm::vec3(-1, -1, -1));
        float lightIntensity = glm::max(glm::dot(normal, -lightDir), 0.0f); // == cos(angle)

        glm::vec3 sphereColor = closestSphere->Albedo;
        sphereColor *= lightIntensity;
        return {sphereColor, 1.0f};
    }

    __device__ glm::u8vec4 ConvertToRGBA(const glm::vec4 &color) {
        return static_cast<glm::u8vec4>(color * 255.f);
    }

}
