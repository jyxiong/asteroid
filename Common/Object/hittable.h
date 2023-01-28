#pragma once

#include <memory>
#include <utility>
#include "Common/ray.h"
#include "Common/vec3.h"
#include "Common/aabb.h"
#include "Common/rng.h"

class material; // declare material without define

struct hit_record {
    point3 p; // 碰撞点坐标
    double t{}; // 碰撞点时间
    vec3 normal; // 碰撞点法向量，本系列中定义法向量与光线方向大致相反
    bool front_face; // 碰撞点是否在外表面
    std::shared_ptr<material> mat_ptr; // 碰撞点材质指针
    double u, v;

    inline void set_face_normal(const ray &r, const vec3 &outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0.0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

class hittable {
public:
    virtual bool hit(const ray &r, double t_min, double t_max, hit_record &rec) const = 0;
    virtual bool bounding_box(double time0, double time1, aabb &output_box) const = 0;
};
