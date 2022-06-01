#pragma once

#include <cmath>

namespace util {

    constexpr double infinity = std::numeric_limits<double>::infinity();
    constexpr double pi = 3.1415926535897932385;

    inline double degrees_to_radians(double degree) {
        return degree * pi / 180.0;
    }

    inline double schlick(double cosine, double ir) {
        // Schlick approximation
        auto r0 = (1 - ir) / (1.0 + ir);
        r0 = r0 * r0;
        return r0 + (1 - r0) * pow(1 - cosine, 5);
    }

    inline double clamp(double x, double min, double max) {
        if (x > max) return max;
        if (x < min) return min;
        return x;
    }
}
