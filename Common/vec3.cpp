#include "vec3.h"

// reflect vector
vec3 reflect(const vec3 &v, const vec3 &n) {
    return v - 2 * dot(n, v) * n;
}

// refract vector
vec3 refract(const vec3 &v, const vec3 &n, double etai_over_etat) {
    auto cos_theta = fmin(dot(-v, n), 1.0);
    vec3 r_out_perp = etai_over_etat * (v + cos_theta * n);
    vec3 r_out_parallel = -sqrt(fabs(1.0 - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}