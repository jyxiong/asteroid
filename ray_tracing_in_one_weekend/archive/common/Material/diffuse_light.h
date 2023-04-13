#pragma once

#include "Common/Texture/solid_color.h"
#include "material.h"

class diffuse_light : public material {
public:
    explicit diffuse_light(std::shared_ptr<texture> emit);
    explicit diffuse_light(const color &emit);

    bool scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered) const override;
    color emitted(double u, double v, const point3 &p) const override;
private:
    std::shared_ptr<texture> m_emit;
};