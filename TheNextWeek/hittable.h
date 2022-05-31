#pragma once

#include <memory>
#include <utility>
#include "Common/ray.h"
#include "Common/vec3.h"
#include "Common/aabb.h"
#include "Common/rtweekend.h"

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

class translate : public hittable {
public:
    translate(std::shared_ptr<hittable> object_ptr, const vec3 &offset)
        : m_object_ptr(std::move(object_ptr)), m_offset(offset) {}

    bool hit(const ray &r, double t_min, double t_max, hit_record &rec) const override;
    bool bounding_box(double time0, double time1, aabb &output_box) const override;
public:
    std::shared_ptr<hittable> m_object_ptr;
    vec3 m_offset;
};

bool translate::hit(const ray &r, double t_min, double t_max, hit_record &rec) const {
    ray moved_r(r.origin() - m_offset, r.direction(), r.time());
    if (!m_object_ptr->hit(moved_r, t_min, t_max, rec))
        return false;

    rec.p += m_offset;
    rec.set_face_normal(moved_r, rec.normal);
    return true;
}

bool translate::bounding_box(double time0, double time1, aabb &output_box) const {
    if (!m_object_ptr->bounding_box(time0, time1, output_box))
        return false;

    output_box = aabb(output_box.min() + m_offset, output_box.max() + m_offset);
    return true;
}

class rotate_y : public hittable {
public:
    rotate_y(std::shared_ptr<hittable> object_ptr, double angle);

    bool hit(const ray &r, double t_min, double t_max, hit_record &rec) const override;
    bool bounding_box(double time0, double time1, aabb &output_box) const override {
        output_box = m_box;
        return m_has_box;
    }

public:
    std::shared_ptr<hittable> m_object_ptr;
    double m_sin_theta;
    double m_cos_theta;
    bool m_has_box;
    aabb m_box;
};

rotate_y::rotate_y(std::shared_ptr<hittable> object_ptr, double angle) : m_object_ptr(std::move(object_ptr)) {
    auto radians = degrees_to_radians(angle);
    m_sin_theta = sin(radians);
    m_cos_theta = cos(radians);
    m_has_box = m_object_ptr->bounding_box(0, 1, m_box);

    point3 min(infinity, infinity, infinity);
    point3 max(-infinity, -infinity, -infinity);

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                auto x = i * m_box.max().x() + (1 - i) * m_box.min().x();
                auto y = j * m_box.max().y() + (1 - j) * m_box.min().y();
                auto z = k * m_box.max().z() + (1 - k) * m_box.min().z();

                auto new_x = m_cos_theta * x + m_sin_theta * z;
                auto new_z = -m_sin_theta * x + m_cos_theta * z;

                vec3 tester(new_x, y, new_z);

                for (int c = 0; c < 3; c++) {
                    min[c] = fmin(min[c], tester[c]);
                    max[c] = fmax(max[c], tester[c]);
                }
            }
        }
    }

    m_box = aabb(min, max);
}

bool rotate_y::hit(const ray &r, double t_min, double t_max, hit_record &rec) const {
    auto origin = r.origin();
    auto direction = r.direction();

    origin[0] = m_cos_theta * r.origin()[0] - m_sin_theta * r.origin()[2];
    origin[2] = m_sin_theta * r.origin()[0] + m_cos_theta * r.origin()[2];

    direction[0] = m_cos_theta * r.direction()[0] - m_sin_theta * r.direction()[2];
    direction[2] = m_sin_theta * r.direction()[0] + m_cos_theta * r.direction()[2];

    ray rotated_r(origin, direction, r.time());

    if (!m_object_ptr->hit(rotated_r, t_min, t_max, rec))
        return false;

    auto p = rec.p;
    auto normal = rec.normal;

    p[0] = m_cos_theta * rec.p[0] + m_sin_theta * rec.p[2];
    p[2] = -m_sin_theta * rec.p[0] + m_cos_theta * rec.p[2];

    normal[0] = m_cos_theta * rec.normal[0] + m_sin_theta * rec.normal[2];
    normal[2] = -m_sin_theta * rec.normal[0] + m_cos_theta * rec.normal[2];

    rec.p = p;
    rec.set_face_normal(rotated_r, normal);

    return true;
}