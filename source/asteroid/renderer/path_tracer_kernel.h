#pragma once

#include <cuda_runtime.h>
#include "glm/glm.hpp"

#include "asteroid/renderer/ray.h"
#include "asteroid/renderer/camera.h"
#include "asteroid/renderer/scene.h"

namespace Asteroid {

    __device__ float HitSphere(const glm::vec3 &center, float radius, const Ray &r) {
        glm::vec3 oc = r.Origin - center;
        auto a = glm::dot(r.Direction, r.Direction);
        auto b = 2.f * glm::dot(oc, r.Direction);
        auto c = dot(oc, oc) - radius * radius;
        auto discriminant = b * b - 4.f * a * c;
        if (discriminant < 0.f) {
            return -1.f;
        } else {
            return (-b - sqrtf(discriminant)) / (2.f * a);
        }
    }

    __device__ glm::vec4 TraceRay(const SceneView &scene, const Ray &ray) {

        if (scene.deviceSpheres.size() == 0)
            return glm::vec4(0);

        int closestSphere = -1;
        float hitDistance = std::numeric_limits<float>::max();

        for (size_t i = 0; i < scene.deviceSpheres.size(); ++i) {
            float t = HitSphere(scene.deviceSpheres[i].Position, scene.deviceSpheres[i].Radius, ray);
            if (t < 0.f) continue;

            if (t < hitDistance) {
                hitDistance = t;
                closestSphere = i;
            }
        }

        if (closestSphere == -1)
            return glm::vec4(0);

        glm::vec3 hitPoint = ray(hitDistance);
        glm::vec3 normal = glm::normalize(hitPoint - scene.deviceSpheres[closestSphere].Position);

        glm::vec3 lightDir = glm::normalize(glm::vec3(-1, -1, -1));
        float lightIntensity = glm::max(glm::dot(normal, -lightDir), 0.0f);

        glm::vec3 sphereColor = scene.deviceSpheres[closestSphere].Albedo;
        sphereColor *= lightIntensity;
        return {sphereColor, 1.0f};
    }

    __device__ glm::u8vec4 ConvertToRGBA(const glm::vec4 &color) {
        return static_cast<glm::u8vec4>(color * 255.f);
    }

}
