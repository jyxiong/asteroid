#pragma once

#include <cuda_runtime.h>
#include "glm/glm.hpp"

#include "asteroid/renderer/scene_struct.h"

namespace Asteroid {

    inline __device__ bool intersect_sphere(const Geometry &geometry, const Ray &r, Intersection &its) {
        auto center = glm::vec3(0);
        auto radius = 1.0f;

        auto origin = glm::vec3(geometry.transform * glm::vec4(r.origin, 1.0f));
        auto direction = glm::vec3(geometry.transform * glm::vec4(r.direction, 0.0f));

        auto oc = origin - center;
        auto a = glm::dot(direction, direction);
        auto half_b = glm::dot(oc, direction);
        auto c = dot(oc, oc) - radius * radius;

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

        float t;
        if (t1 > 0 && t2 > 0) {
            t = glm::min(t1, t2);
        } else {
            t = glm::max(t1, t2);
        }

        auto position = origin + t * direction;

        auto outward_normal = glm::normalize(glm::vec3(geometry.inverseTranspose * glm::vec4(position, 0.0f)));
        its.front_face = glm::dot(direction, outward_normal);
        its.normal = its.front_face ? outward_normal : -outward_normal;

        its.position = glm::vec3(geometry.transform * glm::vec4(position, 1.0f));
        its.t = glm::distance(r.origin, its.position);
        
        its.materialIndex = geometry.materialIndex;

        return true;
    }

}
