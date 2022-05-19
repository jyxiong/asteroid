#pragma once

#include "Utility/hittable.h"
#include "Utility/ray.h"
#include "Utility/vec3.h"

class sphere : public hittable {
public:
    sphere() = default;
    sphere(const point3& center, double radius) : m_center(center), m_radius(radius) {}

    bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const override;
private:
    point3 m_center;
    double m_radius;
};

// https://raytracing.github.io/books/RayTracingInOneWeekend.html#addingasphere/ray-sphereintersection
// hit test with quadratic equation:
// direction⋅direction * t^2 + 2direction⋅(origin−center) * t + (origin−center)⋅(origin−center) − r2 = 0
// discriminant = b * b - 4 * a * c > 0
// root = (-b - sqrt(discriminant)) / (2 * a)
// https://raytracing.github.io/books/RayTracingInOneWeekend.html#surfacenormalsandmultipleobjects/simplifyingtheray-sphereintersectioncode
// simplify the equation by dividing both the numerator and denominator of 2
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
    if (root < t_min || root > t_max)
    {
        root = (-half_b + sqrt_d) / a;
        if (root < t_min || root > t_max)
            return false;
    }

    rec.t = root;
    rec.p = r.at(root);
    // https://raytracing.github.io/books/RayTracingInOneWeekend.html#surfacenormalsandmultipleobjects/frontfacesversusbackfaces
    vec3 outward_normal = (rec.p - m_center) / m_radius;
    rec.set_face_normal(r, outward_normal);

    return true;
}
