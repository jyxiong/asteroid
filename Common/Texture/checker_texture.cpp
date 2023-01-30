#include "checker_texture.h"
#include "Common/vec3.h"
#include "solid_color.h"

checker_texture::checker_texture(std::shared_ptr<texture> even, std::shared_ptr<texture> odd)
    : m_even(std::move(even)),
      m_odd(std::move(odd)) {}

checker_texture::checker_texture(const color &c1, const color &c2)
    : m_even(std::make_shared<solid_color>(c1)),
      m_odd(std::make_shared<solid_color>(c2)) {}

color checker_texture::value(double u, double v, const point3 &p) const {
    // 根据坐标余弦的正负号设置采样纹理
    auto sines = sin(10 * p.x()) * sin(10 * p.y()) * sin(10 * p.z());
    if (sines < 0)
        return m_odd->value(u, v, p);
    else
        return m_even->value(u, v, p);
}