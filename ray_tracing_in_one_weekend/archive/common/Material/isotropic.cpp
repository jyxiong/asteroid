#include "isotropic.h"

isotropic::isotropic(color albedo)
    : m_albedo(std::make_shared<solid_color>(albedo)) {}

isotropic::isotropic(std::shared_ptr<texture> albedo)
    : m_albedo(std::move(albedo)) {}

bool isotropic::scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered) const {
    scattered = ray(rec.p, random_in_unit_sphere(), r_in.time());
    attenuation = m_albedo->value(rec.u, rec.v, rec.p);

    return true;
}