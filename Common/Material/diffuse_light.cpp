#include "diffuse_light.h"

diffuse_light::diffuse_light(std::shared_ptr<texture> emit)
    : m_emit(std::move(emit)) {}

diffuse_light::diffuse_light(const color &emit)
    : m_emit(std::make_shared<solid_color>(emit)) {}

bool diffuse_light::scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered) const {
    return false;
}

color diffuse_light::emitted(double u, double v, const point3 &p) const {
    return m_emit->
        value(u, v, p
    );
}