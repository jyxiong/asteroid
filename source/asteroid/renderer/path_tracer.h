#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "glm/glm.hpp"
#include "asteroid/renderer/camera.h"
#include "asteroid/renderer/scene.h"
#include "asteroid/renderer/path_tracer_kernel.h"

namespace Asteroid {
    __global__ void GeneratePrimaryRay(const Camera camera, BufferView<PathSegment> paths) {
        auto x = blockIdx.x * blockDim.x + threadIdx.x;
        auto y = blockIdx.y * blockDim.y + threadIdx.y;

        auto viewport = camera.viewport;

        if (x >= viewport.x && y >= viewport.y)
            return;

        auto &path = paths[y * viewport.x + x];
        path.color = glm::vec3(0);
        path.throughput = glm::vec3(1);

        auto uv = glm::vec2(x, y) / glm::vec2(viewport) * 2.f - 1.f;

        auto offsetx = float(uv.x) * camera.tanHalfFov * camera.aspectRatio * camera.right;
        auto offsety = float(uv.y) * camera.tanHalfFov * camera.up;

        path.ray.Direction = glm::normalize(camera.direction + offsetx + offsety);
        path.ray.Origin = camera.position;
    }

    __global__ void
    ComputeIntersection(const SceneView scene, const BufferView<PathSegment> paths, int width, int height,
                        BufferView<Intersection> intersections) {
        auto x = blockIdx.x * blockDim.x + threadIdx.x;
        auto y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width && y >= height)
            return;

        const auto &path = paths[y * width + x];

        int closestSphere = -1;
        Intersection its;
        float hitDistance = std::numeric_limits<float>::max();
        for (size_t i = 0; i < scene.deviceSpheres.size(); ++i) {
            if (!HitSphere(scene.deviceSpheres[i], path.ray, its)) continue;

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

    __global__ void
    Shading(const SceneView scene, BufferView<PathSegment> paths, const BufferView<Intersection> its,
            int width, int height) {
        auto x = blockIdx.x * blockDim.x + threadIdx.x;
        auto y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width && y >= height)
            return;

        auto &path = paths[y * width + x];

        auto it = its[y * width + x];

        if (it.t < 0) {
            path.remainingBounces = 0;

            return;
        }

        scatterRay(path, it, scene.deviceMaterials[it.materialId]);
    }

    __global__ void
    ConvertToRGBA(const BufferView<PathSegment> paths, int width, int height, BufferView<glm::u8vec4> image) {
        auto x = blockIdx.x * blockDim.x + threadIdx.x;
        auto y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width && y >= height)
            return;

        auto color = glm::clamp(paths[y * width + x].color, 0.f, 1.f);
        image[y * width + x] = glm::u8vec4(color * 255.f, 255);
    }

}