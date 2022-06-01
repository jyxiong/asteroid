#pragma once

#include "material.h"

class dielectric : public material {
public:
    explicit dielectric(double ir);

    bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered) const override;
private:
    double m_ir; // index of refraction
};