#pragma once

#include <cuda_runtime.h>
#include "glm/glm.hpp"

#include "asteroid/renderer/scene_struct.h"

namespace Asteroid
{

inline __device__ bool intersectSphere(const Geometry& geometry, const Ray& r, Intersection& its)
{
    auto center = glm::vec3(0);
    auto radius = 1.0f;

    auto origin = glm::vec3(geometry.inverseTransform * glm::vec4(r.origin, 1.0f));
    auto direction = glm::vec3(geometry.inverseTransform * glm::vec4(r.direction, 0.0f));

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

    if (t1 < 0 && t2 < 0)
    {
        return false;
    }

    float t;
    if (t1 > 0 && t2 > 0)
    {
        t = glm::min(t1, t2);
    } else
    {
        t = glm::max(t1, t2);
    }

    auto position = origin + t * direction;

    its.normal = glm::normalize(glm::vec3(geometry.inverseTranspose * glm::vec4(position, 0.0f)));

    its.position = glm::vec3(geometry.transform * glm::vec4(position, 1.0f));
    its.t = glm::distance(r.origin, its.position);

    its.materialIndex = geometry.materialIndex;

    return true;
}

inline __device__ bool intersectCube(const Geometry& geometry, const Ray& r, Intersection& its)
{
    auto bot = glm::vec3(-1.0f);
    auto top = glm::vec3(1.0f);

    auto origin = glm::vec3(geometry.inverseTransform * glm::vec4(r.origin, 1.0f));
    auto direction = glm::vec3(geometry.inverseTransform * glm::vec4(r.direction, 0.0f));
    auto inv_direction = 1.0f / direction;

    float tmin = -100000;
    float tmax = 100000;

    glm::vec3 tmin_n, tmax_n;
    float t1, t2;
    float ta, tb;
    for (int i = 0; i < 3; ++i)
    {
        t1 = (bot[i] - origin[i]) * inv_direction[i];
        t2 = (top[i] - origin[i]) * inv_direction[i];
        auto n = glm::vec3(0);

        if (t1 < t2)
        {
            ta = t1;
            tb = t2;
            n[i] = -1.0f;
        } else
        {
            tb = t1;
            ta = t2;
            n[i] = 1.0f;
        }
        if (ta > 0.0f && ta > tmin)
        {
            tmin = ta;
            tmin_n = n;
        }
        if (tb < tmax)
        {
            tmax = tb;
            tmax_n = n;
        }
    }

    // 没有交点
    if (tmin >= tmax || tmax < 0.0f)
    {
        return false;
    }

    // 只有一个交点
    if (tmin <= 0.0f)
    {
        tmin = tmax;
        tmin_n = tmax_n;
        its.front_face = false;
    } else
    {
        its.front_face = true;
    }

    its.normal = glm::normalize(glm::vec3(geometry.inverseTranspose * glm::vec4(tmin_n, 0.0f)));
    its.position = glm::vec3(geometry.transform * glm::vec4(r.origin + tmin * r.direction, 1.0f));
    its.t = glm::distance(r.origin, its.position);

    its.materialIndex = geometry.materialIndex;

    return true;
}

inline __device__ bool intersectSquare(const Geometry& geometry, const Ray& r, Intersection& its)
{
    auto origin = glm::vec3(geometry.inverseTransform * glm::vec4(r.origin, 1.0f));
    auto direction = glm::vec3(geometry.inverseTransform * glm::vec4(r.direction, 0.0f));

    return true;
}

} // namespace Asteroid
