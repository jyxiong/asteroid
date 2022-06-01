#include "metal.h"

metal::metal(const color &albedo, double fuzzy)
    : m_albedo(albedo), m_fuzzy(fuzzy < 1.0 ? fuzzy : 1.0) {}

bool metal::scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered) const {
    // metal material with reflect direction
    vec3 reflect_direction = reflect(r_in.direction(), rec.normal);

    scattered = ray(rec.p, reflect_direction + m_fuzzy * random_in_unit_sphere(), r_in.time());
    attenuation = m_albedo;

    return dot(reflect_direction, rec.normal) > 0.0;
}