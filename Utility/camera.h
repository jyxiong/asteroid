#pragma once

#include "vec3.h"
#include "ray.h"

class camera {
public:
    camera(
        point3 lookfrom,
        point3 lookat,
        vec3 vup,
        double vfov, // vertical field-of-view in degrees
        double aspect_ratio
    ) {
        auto theta = degrees_to_radians(vfov);
        auto h = tan(theta / 2.0);
        auto viewport_height = 2.0 * h;
        auto viewport_width = aspect_ratio * viewport_height;

        auto w = unit_vector(lookfrom - lookat);
        auto u = unit_vector(cross(vup, w));
        auto v = cross(w, u);

        m_origin = lookfrom;
        m_horizontal = viewport_width * u;
        m_vertical = viewport_height * v;
        m_left_lower_corner = m_origin - m_horizontal / 2 - m_vertical / 2 - w;
    }

    ray get_ray(double s, double t) {
        return ray{m_origin, m_left_lower_corner + s * m_horizontal + t * m_vertical - m_origin};
    }

private:
    point3 m_origin;
    vec3 m_horizontal;
    vec3 m_vertical;
    point3 m_left_lower_corner;
};
