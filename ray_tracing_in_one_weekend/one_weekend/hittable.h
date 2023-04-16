#pragma once

#include "glm/glm.hpp"

struct HitRecord
{
    glm::vec3 p;
    glm::vec3 normal;
    float t;
    bool front_face;

    // 设置法向量朝外
    inline __device__ void set_face_normal(const Ray &ray, const glm::vec3 &outward_normal)
    {
        // 法向量与射线方向点乘，如果小于0，说明法向量与射线方向相反
        // 即法线永远从交点出射，所以叫outward_normal
        front_face = glm::dot(ray.direction, outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

enum class HittableType
{
    Sphere
};

class Hittable
{
public:
    HittableType type;

public:
    __device__ virtual bool hit(const Ray &ray, float t_min, float t_max, HitRecord &rec) const = 0;
};
