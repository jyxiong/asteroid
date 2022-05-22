#pragma once

#include <limits>
#include <random>

constexpr double infinity = std::numeric_limits<double>::infinity();
constexpr double pi = 3.1415926535897932385;

inline double random_double()
{
    static std::uniform_real_distribution<double> distribution(0.0, 1.0);
    static std::mt19937 generator;
    return distribution(generator);
}

inline double random_double(double min, double max)
{
    return min + (max - min) * random_double();
}

inline vec3 random_vec3()
{
    return vec3{
        random_double(),
        random_double(),
        random_double()
    };
}

inline vec3 random_vec3(double min, double max)
{
    return vec3{
        random_double(min, max),
        random_double(min, max),
        random_double(min, max)
    };
}

vec3 random_in_unit_sphere()
{
    while (true)
    {
        vec3 p = random_vec3(-1.0, 1.0);
        if (p.length_squared() >= 1.0) continue;
        return p;
    }
}
vec3 random_unit_vector()
{
    return unit_vector(random_in_unit_sphere());
}

vec3 random_in_hemisphere(const vec3& normal)
{
    vec3 in_unit_sphere = random_in_unit_sphere();
    if (dot(in_unit_sphere, normal) > 0.0)
        return in_unit_sphere;
    else
        return -in_unit_sphere;
}

inline double clamp(double x, double min, double max)
{
    if (x > max) return max;
    if (x < min) return min;
    return x;
}

inline double degrees_to_radians(double degree)
{
    return degree * pi / 180.0;
}