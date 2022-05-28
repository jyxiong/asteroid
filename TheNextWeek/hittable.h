#pragma once

#include <memory>
#include "Common/ray.h"
#include "Common/vec3.h"
#include "Common/aabb.h"

class material; // declare material without define

struct hit_record {
    point3 p;
    double t{};
    vec3 normal;
    bool front_face;
    std::shared_ptr<material> mat_ptr;
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
