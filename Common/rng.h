#pragma once

#include <random>

#include "vec3.h"

inline double random_double() {
    static std::uniform_real_distribution<double> distribution(0.0, 1.0);
    static std::mt19937 generator;
    return distribution(generator);
}

inline double random_double(double min, double max) {
    return min + (max - min) * random_double();
}

inline int random_int(int min, int max) {
    return static_cast<int>(random_double(min, max));
}

inline vec3 random_vec3() {
    return vec3{
        random_double(),
        random_double(),
        random_double()
    };
}

inline vec3 random_vec3(double min, double max) {
    return vec3{
        random_double(min, max),
        random_double(min, max),
        random_double(min, max)
    };
}

inline vec3 random_in_unit_sphere() {
    while (true) {
        vec3 p = random_vec3(-1.0, 1.0);
        if (p.length_squared() >= 1.0) continue;
        return p;
    }
}
inline vec3 random_unit_vector() {
    return unit_vector(random_in_unit_sphere());
}

inline vec3 random_in_hemisphere(const vec3 &normal) {
    vec3 in_unit_sphere = random_in_unit_sphere();
    if (dot(in_unit_sphere, normal) > 0.0)
        return in_unit_sphere;
    else
        return -in_unit_sphere;
}

inline vec3 random_in_unit_disk() {
    while (true) {
        auto p = vec3(random_double(-1, 1), random_double(-1, 1), 0);
        if (p.length_squared() >= 1.0) continue;
        return p;
    }
}
