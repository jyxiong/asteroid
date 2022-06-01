#include "dielectric.h"
#include <cmath>
#include "Common/utils.h"

dielectric::dielectric(double ir)
    : m_ir(ir) {}

bool dielectric::scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered) const {
// front_face: from air(1.0) to dielectric(m_ir)
    double refraction_ratio = rec.front_face ? (1.0 / m_ir) : m_ir;

    auto in_direction = unit_vector(r_in.direction());
    auto cos_theta = fmin(dot(-in_direction, rec.normal), 1.0);
    auto sin_theta = sqrt(1 - cos_theta * cos_theta);

    vec3 scatter_direction;
    bool cannot_refract = refraction_ratio * sin_theta > 1.0;
    if (cannot_refract || util::schlick(cos_theta, refraction_ratio) > random_double()) // maybe not refraction
        scatter_direction = reflect(in_direction, rec.normal);
    else
        scatter_direction = refract(in_direction, rec.normal, refraction_ratio);

    scattered = ray(rec.p, scatter_direction, r_in.time());
    attenuation = color{1.0, 1.0, 1.0};

    return true;
}