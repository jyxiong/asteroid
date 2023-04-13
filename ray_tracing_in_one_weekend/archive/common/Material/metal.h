#pragma once

#include "material.h"

class metal : public material {
public:
    explicit metal(const color &albedo, double fuzzy);

    bool scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered) const override;
private:
    color m_albedo;
    double m_fuzzy;
};