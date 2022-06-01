#pragma once

#include <memory>
#include "Common/vec3.h"
#include "texture.h"

class checker_texture : public texture {
public:
    checker_texture() = default;
    checker_texture(std::shared_ptr<texture> even, std::shared_ptr<texture> odd);
    checker_texture(const color &c1, const color &c2);

    color value(double u, double v, const point3 &p) const override;
private:
    std::shared_ptr<texture> m_even;
    std::shared_ptr<texture> m_odd;
};