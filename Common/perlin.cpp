#include "perlin.h"
#include <cmath>
#include "rng.h"

perlin::perlin() {
    for (int i = 0; i < POINT_COUNT; ++i) {
        m_ran_vec[i] = unit_vector(random_vec3(-1, 1));
        m_perm_x[i] = i;
        m_perm_y[i] = i;
        m_perm_z[i] = i;
    }

    permute(m_perm_x.data(), POINT_COUNT);
    permute(m_perm_y.data(), POINT_COUNT);
    permute(m_perm_z.data(), POINT_COUNT);
}

double perlin::noise(const point3 &p) const {
    // get decimal of p
    auto u = p.x() - floor(p.x());
    auto v = p.y() - floor(p.y());
    auto w = p.z() - floor(p.z());

    // get integer of p
    auto i = static_cast<int>(floor(p.x()));
    auto j = static_cast<int>(floor(p.y()));
    auto k = static_cast<int>(floor(p.z()));
    vec3 c[2][2][2];

    for (int di = 0; di < 2; di++)
        for (int dj = 0; dj < 2; dj++)
            for (int dk = 0; dk < 2; dk++)
                c[di][dj][dk] = m_ran_vec[
                    m_perm_x[(i + di) & 255] ^ m_perm_y[(j + dj) & 255] ^ m_perm_z[(k + dk) & 255]
                ];
    // smooth the noise by trilinear interpolate
    return perlin_interp(c, u, v, w);
}

double perlin::turb(const point3 &p, int depth) const {
    auto accum = 0.0;
    auto temp_p = p;
    auto weight = 1.0;

    for (int i = 0; i < depth; i++) {
        accum += weight * noise(temp_p);
        weight *= 0.5;
        temp_p *= 2;
    }

    return fabs(accum);
}

void perlin::permute(int *p, int n) {
    for (int i = n - 1; i > 0; i--) {
        int target = random_int(0, i);
        std::swap(p[i], p[target]);
    }
}

double perlin::perlin_interp(vec3 c[2][2][2], double u, double v, double w) {
    auto uu = u * u * (3 - 2 * u);
    auto vv = v * v * (3 - 2 * v);
    auto ww = w * w * (3 - 2 * w);
    auto accum = 0.0;

    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++) {
                vec3 weight_v(u - i, v - j, w - k);
                accum += (i * uu + (1 - i) * (1 - uu)) *
                    (j * vv + (1 - j) * (1 - vv)) *
                    (k * ww + (1 - k) * (1 - ww)) *
                    dot(c[i][j][k], weight_v);
            }

    return accum;
}