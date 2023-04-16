#pragma once

#include "glm/glm.hpp"

#include "helper_cuda.h"

#include "hittable.h"
#include "sphere.h"
#include "hittable_list.h"

__device__ float hit_sphere(const glm::vec3 &center, float radius, const Ray &ray)
{
    auto oc = ray.origin - center;
    auto a = glm::length(ray.direction);
    auto half_b = glm::dot(oc, ray.direction);
    auto c = glm::length(oc) - radius * radius;
    auto discriminant = half_b * half_b - a * c;
    if (discriminant < 0)
        // 没有交点
        return -1.f;
    else
        // 有交点，返回最近的交点
        return (-half_b - sqrt(discriminant)) / a;
}

__device__ glm::vec3 ray_color(const Ray &ray, Hittable **world)
{
    HitRecord rec;
    if ((*world)->hit(ray, 0.0, FLT_MAX, rec))
    {
        // 有交点，返回法向量颜色
        return 0.5f * (rec.normal + 1.f);
    }

    // 没有交点，返回背景色
    auto unit_vector = glm::normalize(ray.direction);
    float t = (unit_vector.y + 1.f) * 0.5f;
    return glm::mix(glm::vec3(1.f), glm::vec3(0.5, 0.7, 1.0), t);
}

__global__ void render(unsigned int width, unsigned int height, Camera *camera, Hittable **world, glm::u8vec3 *fb)
{
    auto i = threadIdx.x + blockIdx.x * blockDim.x;
    auto j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i >= width || j >= height) return;

    auto ray = camera->generate_ray(i, j);

    fb[j * width + i] = ray_color(ray, world) * 255.f;
}
