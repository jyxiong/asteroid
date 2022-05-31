#pragma once

#include <memory>
#include "hittable.h"
#include "Common/ray.h"
#include "Common/vec3.h"

class sphere : public hittable {
public:
    sphere() = default;
    sphere(const point3 &center, double radius) : m_center(center), m_radius(radius) {}
    sphere(const point3 &center, double radius, std::shared_ptr<material> mat_ptr)
        : m_center(center), m_radius(radius), m_mat_ptr(std::move(mat_ptr)) {}

    bool hit(const ray &r, double t_min, double t_max, hit_record &rec) const override;
private:
    point3 m_center;
    double m_radius{};
    std::shared_ptr<material> m_mat_ptr;
};

bool sphere::hit(const ray &r, double t_min, double t_max, hit_record &rec) const {
    vec3 oc = r.origin() - m_center;
    auto a = r.direction().length_squared();
    auto half_b = dot(r.direction(), oc);
    auto c = oc.length_squared() - m_radius * m_radius;
    auto discriminant = half_b * half_b - a * c;

    if (discriminant < 0.0)
        return false;

    auto sqrt_d = sqrt(discriminant);
    auto root = (-half_b - sqrt_d) / a;
    if (root < t_min || root > t_max) {
        root = (-half_b + sqrt_d) / a;
        if (root < t_min || root > t_max)
            return false;
    }

    rec.t = root;
    rec.p = r.at(root);
    vec3 outward_normal = (rec.p - m_center) / m_radius;
    rec.set_face_normal(r, outward_normal);
    rec.mat_ptr = m_mat_ptr;

    return true;
}
