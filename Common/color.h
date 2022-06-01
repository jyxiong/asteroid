#pragma once

#include <vector>
#include <ostream>
#include "vec3.h"
#include "utils.h"

void write_color(const color &pixel_color, int samples_per_pixel, int pixel_index, std::vector<unsigned char>& image) {
    auto r = pixel_color.x();
    auto g = pixel_color.y();
    auto b = pixel_color.z();

    auto scale = 1.0 / samples_per_pixel;

    r = sqrt(r * scale);
    g = sqrt(g * scale);
    b = sqrt(b * scale);

    image[pixel_index * 3 + 0] = static_cast<unsigned char>(256 * util::clamp(r, 0.0, 0.999));
    image[pixel_index * 3 + 1] = static_cast<unsigned char>(256 * util::clamp(g, 0.0, 0.999));
    image[pixel_index * 3 + 2] = static_cast<unsigned char>(256 * util::clamp(b, 0.0, 0.999));
}

