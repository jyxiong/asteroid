#pragma once

#include "Common/vec3.h"
#include "Common/perlin.h"
#include "texture.h"

class noise_texture : public texture {
public:
    noise_texture() = default;
    explicit noise_texture(double scale);

    color value(double u, double v, const point3 &p) const override;

public:
    perlin m_noise;
    double m_scale{};
};