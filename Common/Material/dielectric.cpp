#include "dielectric.h"
#include <cmath>
#include "Common/utils.h"

dielectric::dielectric(double ir)
    : m_ir(ir) {}

bool dielectric::scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered) const {
    // front_face：从空气(ior=1.0)射入介电质(m_ir)
    double refraction_ratio = rec.front_face ? (1.0 / m_ir) : m_ir;

    auto in_direction = unit_vector(r_in.direction());
    auto cos_theta = fmin(dot(-in_direction, rec.normal), 1.0);
    auto sin_theta = sqrt(1 - cos_theta * cos_theta);

    // snell折射定律：当光线从折射率大的介质进入折射率小的介质时，如果入射角过大，则不会发生折射，而发生反射
    // fresnel方程：当观察角度/入射角度与物体表面平行时，大部分光线会被反射；一般采用schlick近似计算
    vec3 scatter_direction;
    bool cannot_refract = refraction_ratio * sin_theta > 1.0;
    if (cannot_refract || util::schlick(cos_theta, refraction_ratio) > random_double())
        scatter_direction = reflect(in_direction, rec.normal);
    else
        scatter_direction = refract(in_direction, rec.normal, refraction_ratio);

    scattered = ray(rec.p, scatter_direction, r_in.time());

    // 不衰减
    attenuation = color{1.0, 1.0, 1.0};

    return true;
}