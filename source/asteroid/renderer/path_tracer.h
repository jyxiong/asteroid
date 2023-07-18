#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "glm/glm.hpp"
#include "asteroid/renderer/camera.h"
#include "asteroid/renderer/scene.h"
#include "asteroid/renderer/path_tracer_kernel.h"

namespace Asteroid {
    __global__ void GeneratePrimaryRay(const Camera camera, Ray *rays) {
        auto x = blockIdx.x * blockDim.x + threadIdx.x;
        auto y = blockIdx.y * blockDim.y + threadIdx.y;

        auto viewport = camera.GetViewport();

        if (x >= viewport.x && y >= viewport.y)
            return;

        auto uv = glm::vec2(x, y) / glm::vec2(viewport) * 2.f - 1.f;

        camera.GeneratePrimaryRay(uv, rays[y * viewport.x + x]);
    }

    __global__ void
    ComputeIntersection(const SceneView scene, const Ray *rays, int width, int height, Intersection *intersections) {
        auto x = blockIdx.x * blockDim.x + threadIdx.x;
        auto y = blockIdx.y * blockDim.y + threadIdx.y;

        int closestSphere = -1;
        Intersection its;
        float hitDistance = std::numeric_limits<float>::max();
        for (size_t i = 0; i < scene.deviceSpheres.size(); ++i) {
            if (!HitSphere(scene.deviceSpheres[i], rays[y * width + x], its)) continue;

            if (its.t < hitDistance && its.t > 0) {
                hitDistance = its.t;
                closestSphere = i;
            }
        }

        if (closestSphere < 0)
            intersections[y * width + x].t = -1;
        else {
            intersections[y * width + x] = its;
        }
    }

    __global__ void Shading(const Ray *rays, const Intersection *its, glm::vec4 *g_odata, int width, int height) {
        auto x = blockIdx.x * blockDim.x + threadIdx.x;
        auto y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width && y >= height)
            return;

        float multiplier = 1.0f;

        auto it = its[y * width + x];
        auto ray = rays[y * width + x];
        if (it.t < 0)
        {
            return;
        }

        glm::vec3 lightDir = glm::normalize(glm::vec3(-1, -1, -1));
        float lightIntensity = glm::max(glm::dot(it.normal, -lightDir), 0.0f);

        auto color = it.albedo * lightIntensity;
        g_odata[y * width + x] +=  glm::vec4(color, 1.0f) * multiplier;

        multiplier *= 0.7f;

        ray.Origin = its->position + its->normal * 0.0001f;
        ray.Direction = glm::reflect(ray.Direction, its->normal);
    }

    __global__ void ConvertToRGBA(const glm::vec4* data, int width, int height, glm::u8vec4* image)
    {
        auto x = blockIdx.x * blockDim.x + threadIdx.x;
        auto y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width && y >= height)
            return;

        auto color = glm::clamp(data[y * width + x], 0.f, 1.f);
        image[y * width + x] = glm::u8vec4(color * 255.f);
    }

}