#pragma once

#include "Common/Texture/solid_color.h"
#include "material.h"

class isotropic : public material {
public:
    explicit isotropic(color albedo);
    isotropic(std::shared_ptr<texture> albedo);

    bool scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered) const override;

public:
    std::shared_ptr<texture> m_albedo;
};