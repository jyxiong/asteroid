#include "solid_color.h"

solid_color::solid_color(const color &c) : m_color(c) {}

solid_color::solid_color(double r, double g, double b) : solid_color(color{r, g, b}) {}

color solid_color::value(double u, double v, const vec3 &p) const {
    return m_color;
}

