#pragma once

#include "Common/Texture/solid_color.h"
#include "material.h"

class lambertian : public material {
public:
    explicit lambertian(const color &albedo);
    explicit lambertian(std::shared_ptr<texture> albedo);

    bool scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered) const override;
private:
    std::shared_ptr<texture> m_albedo;
};