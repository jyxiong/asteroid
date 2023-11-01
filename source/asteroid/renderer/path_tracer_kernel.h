#pragma once

#include <cuda_runtime.h>
#include "glm/glm.hpp"

#include "asteroid/renderer/scene.h"
#include "asteroid/renderer/scene_struct.h"

namespace Asteroid {

    __device__
    void scatterRay(PathSegment & pathSegment, const Intersection& its, const Material &mat) {

        auto lightDir = glm::normalize(glm::vec3(-1, -1, -1));
        auto lightIntensity = glm::max(glm::dot(its.normal, -lightDir), 0.0f);

        auto color = mat.albedo * lightIntensity;
        pathSegment.color += color * pathSegment.throughput;

        pathSegment.throughput *= 0.5f;

        pathSegment.ray.origin = its.position + its.normal * 0.0001f;
        pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, its.normal + mat.roughness);

        pathSegment.remainingBounces--;
    }

}
