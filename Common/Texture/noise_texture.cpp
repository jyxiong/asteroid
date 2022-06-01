#include "noise_texture.h"

noise_texture::noise_texture(double scale) : m_scale(scale) {}

color noise_texture::value(double u, double v, const point3 &p) const {
//    return color(1, 1, 1) * 0.5 * (1 + m_noise.noise(m_scale * p));
//    return color(1, 1, 1) * m_noise.turb(m_scale * p);
    return color(1, 1, 1) * 0.5 * (1 + sin(m_scale * p.z() + 10 * m_noise.turb(p)));
}