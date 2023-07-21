#pragma once

#include <cuda_runtime.h>
#include "glm/glm.hpp"

#include "asteroid/renderer/scene.h"
#include "asteroid/renderer/scene_struct.h"

namespace Asteroid {

    __device__ bool HitSphere(const Sphere &sphere, const Ray &r, Intersection &its) {
        glm::vec3 oc = r.Origin - sphere.Position;
        auto a = glm::dot(r.Direction, r.Direction);
        auto half_b = glm::dot(oc, r.Direction);
        auto c = dot(oc, oc) - sphere.Radius * sphere.Radius;

        auto discriminant = half_b * half_b - a * c;
        if (discriminant < 0)
        {
            return false;
        }
        auto sqrtd = sqrt(discriminant);

        auto t1 = (-half_b - sqrtd) / a;
        auto t2 = (-half_b + sqrtd) / a;

        if (t1 < 0 && t2 < 0) {
             return false;
        }

        if (t1 > 0 && t2 > 0) {
            its.t = glm::min(t1, t2);
            its.position = r.Origin + its.t * r.Direction;
            its.normal = (its.position - sphere.Position) / sphere.Radius;
            its.materialId = sphere.MaterialId;
        } else {
            its.t = glm::max(t1, t2);
            its.position = r.Origin + its.t * r.Direction;
            its.normal = (sphere.Position - its.position) / sphere.Radius;
            its.materialId = sphere.MaterialId;
        }

        return true;
    }

    __device__
    void scatterRay(PathSegment & pathSegment, const Intersection& its, const Material &mat) {

        auto lightDir = glm::normalize(glm::vec3(-1, -1, -1));
        auto lightIntensity = glm::max(glm::dot(its.normal, -lightDir), 0.0f);

        auto color = mat.Albedo * lightIntensity;
        pathSegment.color += color * pathSegment.throughput;

        pathSegment.throughput *= 0.5f;

        pathSegment.ray.Origin = its.position + its.normal * 0.0001f;
        pathSegment.ray.Direction = glm::reflect(pathSegment.ray.Direction, its.normal + mat.Roughness);

        pathSegment.remainingBounces--;
    }

}
