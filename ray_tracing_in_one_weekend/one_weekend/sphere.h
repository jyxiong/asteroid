#pragma once

#include "glm/glm.hpp"
#include "hittable.h"

class Sphere : public Hittable
{
public:
    glm::vec3 center;
    float radius{};

public:
    Sphere() = default;

    __host__ __device__ Sphere(const glm::vec3 &center, float radius)
        : center(center), radius(radius) {}

    __device__ bool hit(const Ray &ray, float t_min, float t_max, HitRecord &rec) const override;
};

__device__ bool Sphere::hit(const Ray &ray, float t_min, float t_max, HitRecord &rec) const
{
    auto oc = ray.origin - center;
    auto a = glm::length(ray.direction);
    auto half_b = glm::dot(oc, ray.direction);
    auto c = glm::length(oc) - radius * radius;
    auto discriminant = half_b * half_b - a * c;

    // 没有交点
    if (discriminant < 0)
        return false;

    // 有交点，返回最近的交点
    auto sqrt_d = sqrt(discriminant);

    // 寻找范围内的最近交点
    auto t = (-half_b - sqrt_d) / a;
    if (t < t_min || t > t_max) {
        // 近交点不在范围内，取远交点
        t = (-half_b + sqrt_d) / a;
        if (t < t_min || t > t_max)
            // 远交点也不在范围内，返回false
            return false;
    }

    // 记录交点信息
    rec.t = t;
    rec.p = ray(t);
    // 从球心到交点的向量
    auto outward_normal = (rec.p - center) / radius;
    rec.set_face_normal(ray, outward_normal);

    return true;
}