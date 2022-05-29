#pragma once

#include <memory>
#include <utility>
#include "hittable.h"
#include "Common/ray.h"
#include "Common/vec3.h"
#include "Common/aabb.h"
#include "Common/rtweekend.h"

class sphere : public hittable {
public:
    sphere() = default;
    sphere(const point3& center, double radius) : m_center(center), m_radius(radius) {}
    sphere(const point3& center, double radius, std::shared_ptr<material> mat_ptr)
        : m_center(center), m_radius(radius), m_mat_ptr(std::move(mat_ptr)) {}

    bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const override;
    bool bounding_box(double time0, double time1, aabb& output_box) const override;

private:
    static void get_sphere_uv(const point3& p, double& u, double& v) {
        auto theta = acos(-p.y());
        auto phi = atan2(-p.z(), p.x()) + pi;

        u = phi / (2 * pi);
        v = theta / pi;
    }

private:
    point3 m_center;
    double m_radius{};
    std::shared_ptr<material> m_mat_ptr;
};

bool sphere::hit(const ray& r, double t_min, double t_max, hit_record& rec) const {
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
    get_sphere_uv(outward_normal, rec.u, rec.v);
    rec.mat_ptr = m_mat_ptr;

    return true;
}

bool sphere::bounding_box(double time0, double time1, aabb& output_box) const {
    output_box = aabb(m_center - vec3(m_radius, m_radius, m_radius),
        m_center + vec3(m_radius, m_radius, m_radius));
    return true;
}
