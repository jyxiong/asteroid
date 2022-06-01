#pragma once

#include <array>
#include "vec3.h"

class perlin {
public:
    perlin();
    ~perlin() = default;

    double noise(const point3 &p) const;
    double turb(const point3 &p, int depth = 7) const;

private:
    static void permute(int *p, int n);
    static double perlin_interp(vec3 c[2][2][2], double u, double v, double w);

private:
    static const int POINT_COUNT = 256;
    std::array<vec3, POINT_COUNT> m_ran_vec;
    std::array<int, POINT_COUNT> m_perm_x;
    std::array<int, POINT_COUNT> m_perm_y;
    std::array<int, POINT_COUNT> m_perm_z;
};
