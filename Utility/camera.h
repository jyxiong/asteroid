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
        double aspect_ratio,
        double aperture,
        double focus_dist
    ) {
        auto theta = degrees_to_radians(vfov);
        auto h = tan(theta / 2.0);
        auto viewport_height = 2.0 * h;
        auto viewport_width = aspect_ratio * viewport_height;

        m_w = unit_vector(lookfrom - lookat);
        m_u = unit_vector(cross(vup, m_w));
        m_v = cross(m_w, m_u);

        m_origin = lookfrom;
        m_horizontal = focus_dist * viewport_width * m_u;
        m_vertical = focus_dist * viewport_height * m_v;
        m_left_lower_corner = m_origin - m_horizontal / 2 - m_vertical / 2 - focus_dist * m_w;

        m_lens_radius = aperture / 2;
    }

    ray get_ray(double s, double t) {
        vec3 rd = m_lens_radius * random_in_unit_disk();
        vec3 offset = m_u * rd.x() + m_v * rd.y();

        return ray{ m_origin + offset, m_left_lower_corner + s * m_horizontal + t * m_vertical - m_origin - offset };
    }

private:
    point3 m_origin;
    point3 m_left_lower_corner;
    vec3 m_horizontal;
    vec3 m_vertical;
    vec3 m_u, m_v, m_w;
    double m_lens_radius;
};
