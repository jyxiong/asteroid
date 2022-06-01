#pragma once

#include "Common/vec3.h"
#include "texture.h"

class solid_color : public texture {
public:
    solid_color() = default;
    explicit solid_color(const color &c);
    solid_color(double r, double g, double b);

    color value(double u, double v, const vec3 &p) const override;
private:
    color m_color;
};