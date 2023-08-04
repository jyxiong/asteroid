#pragma once

#include <cuda_runtime.h>
#include "glm/glm.hpp"

#include "asteroid/renderer/scene.h"
#include "asteroid/renderer/scene_struct.h"

namespace Asteroid {

    __device__ bool HitSphere(const Sphere &sphere, const Ray &r, Intersection &its) {
        glm::vec3 oc = r.origin - sphere.position;
        auto a = glm::dot(r.direction, r.direction);
        auto half_b = glm::dot(oc, r.direction);
        auto c = dot(oc, oc) - sphere.radius * sphere.radius;

        auto discriminant = half_b * half_b - a * c;
        if (discriminant < 0)
        {
            return false;
        }
        auto sqrtDis = sqrt(discriminant);

        auto t1 = (-half_b - sqrtDis) / a;
        auto t2 = (-half_b + sqrtDis) / a;

        if (t1 < 0 && t2 < 0) {
             return false;
        }

        if (t1 > 0 && t2 > 0) {
            its.t = glm::min(t1, t2);
            its.position = r.origin + its.t * r.direction;
            its.normal = (its.position - sphere.position) / sphere.radius;
            its.materialIndex = sphere.materialIndex;
        } else {
            its.t = glm::max(t1, t2);
            its.position = r.origin + its.t * r.direction;
            its.normal = (sphere.position - its.position) / sphere.radius;
            its.materialIndex = sphere.materialIndex;
        }

        return true;
    }

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
