#pragma once

#include "vec3.h"
#include "rtweekend.h"

class perlin {
public:
    perlin() {
        m_ran_vec = new vec3[POINT_COUNT];
        for (int i = 0; i < POINT_COUNT; ++i) {
            m_ran_vec[i] = unit_vector(random_vec3(-1, 1));
        }

        m_perm_x = perlin_generate_perm();
        m_perm_y = perlin_generate_perm();
        m_perm_z = perlin_generate_perm();
    }

    ~perlin() {
        delete[] m_ran_vec;
        delete[] m_perm_x;
        delete[] m_perm_y;
        delete[] m_perm_z;
    }

    double noise(const point3 &p) const {
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

    double turb(const point3 &p, int depth = 7) const {
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

private:
    static const int POINT_COUNT = 256;
    vec3 *m_ran_vec; // p[rand(0, 1), rand(0, 1), ..., rand(0, 1)]
    int *m_perm_x; // rand_sort(p[0, 1, ..., 255])
    int *m_perm_y; // rand_sort(p[0, 1, ..., 255])
    int *m_perm_z; // rand_sort(p[0, 1, ..., 255])

    // rand(p[0, 1, ..., 255])
    static int *perlin_generate_perm() {
        auto p = new int[POINT_COUNT];
        // p[0, 1, ..., 255]
        for (int i = 0; i < perlin::POINT_COUNT; i++)
            p[i] = i;

        permute(p, POINT_COUNT);

        return p;
    }

    // swap p[255], p[random(0, 254)]
    // swap p[254], p[random(0, 253)]
    // ...
    // swap p[1], p[0]
    static void permute(int *p, int n) {
        for (int i = n - 1; i > 0; i--) {
            int target = random_int(0, i);
            std::swap(p[i], p[target]);
        }
    }

    static double perlin_interp(vec3 c[2][2][2], double u, double v, double w) {
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
                        (k * ww + (1 - k) * (1 - ww)) * dot(c[i][j][k], weight_v);
                }

        return accum;
    }
};
