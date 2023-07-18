#pragma once

#include <cuda_runtime.h>
#include "glm/glm.hpp"

#include "asteroid/renderer/ray.h"
#include "asteroid/renderer/camera.h"
#include "asteroid/renderer/scene.h"
#include "asteroid/renderer/intersection.h"

namespace Asteroid {

    __device__ bool HitSphere(const Sphere &sphere, const Ray &r, Intersection &its) {
        glm::vec3 oc = r.Origin - sphere.Position;
        auto a = glm::dot(r.Direction, r.Direction);
        auto half_b = glm::dot(oc, r.Direction);
        auto c = dot(oc, oc) - sphere.Radius * sphere.Radius;

        auto discriminant = half_b * half_b - a * c;
        if (discriminant < 0) return false;
        auto sqrtd = sqrt(discriminant);

        auto t1 = (-half_b - sqrtd) / a;
        auto t2 = (-half_b + sqrtd) / a;

        if (t1 < 0 && t2 < 0) {
            its.t = -1;
            return false;
        }

        if (t1 > 0 && t2 > 0) {
            its.t = glm::min(t1, t2);
            its.position = r(its.t);
            its.normal = (its.position - sphere.Position) / sphere.Radius;
            its.albedo = sphere.Albedo;
        } else {
            its.t = glm::max(t1, t2);
            its.position = r(its.t);
            its.normal = (sphere.Position - its.position) / sphere.Radius;
            its.albedo = sphere.Albedo;
        }

        return true;
    }

    __device__ glm::vec4 TraceRay(const SceneView &scene, const Ray &ray, const Intersection &its) {
        if (its.t < 0)
            return glm::vec4(1);

        glm::vec3 lightDir = glm::normalize(glm::vec3(-1, -1, -1));
        float lightIntensity = glm::max(glm::dot(its.normal, -lightDir), 0.0f);

        auto color = its.albedo * lightIntensity;
        return {color, 1.0f};
    }

    __device__ glm::u8vec4 ConvertToRGBA(const glm::vec4 &color) {
        return static_cast<glm::u8vec4>(color * 255.f);
    }

}
