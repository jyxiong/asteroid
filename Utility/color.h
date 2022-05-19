#pragma once

#include <ostream>
#include "vec3.h"
#include "rtweekend.h"

void write_color(const color& pixel_color, int samples_per_pixel, std::ostream& out)
{
    auto color = pixel_color / samples_per_pixel;

    out << static_cast<int>(256 * clamp(color.x(), 0.0, 0.999)) << ' '
        << static_cast<int>(256 * clamp(color.y(), 0.0, 0.999)) << ' '
        << static_cast<int>(256 * clamp(color.z(), 0.0, 0.999)) << '\n';
}

