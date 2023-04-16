#pragma once

#include "hittable.h"

class HittableList : public Hittable
{
public:
    Hittable **list{ nullptr };
    unsigned int list_size{};

public:
    __host__ __device__ HittableList() {};

    __host__ __device__ HittableList(Hittable **list, unsigned int list_size)
        : list(list), list_size(list_size) {}

    __device__ bool hit(const Ray &ray, float t_min, float t_max, HitRecord &rec) const override;
};

__device__ bool HittableList::hit(const Ray &ray, float t_min, float t_max, HitRecord &rec) const
{
    HitRecord temp_rec;
    bool hit_anything = false;
    auto closest_so_far = t_max;

    // 遍历所有物体，寻找最近的交点
    for (unsigned int i = 0; i < list_size; ++i)
    {
        // 如果有交点，且交点距离小于之前的交点，更新交点信息
        if (list[i]->hit(ray, t_min, closest_so_far, temp_rec))
        {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}
