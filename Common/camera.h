#pragma once

#include "vec3.h"
#include "ray.h"
#include "rtweekend.h"

class camera {
public:
    camera(
        point3 look_from,
        point3 look_at,
        vec3 view_up,
        double vertical_fov, // vertical field-of-view in degrees
        double aspect_ratio,
        double aperture,
        double focus_dist,
        double time0 = 0,
        double time1 = 0
    ) {
        auto theta = degrees_to_radians(vertical_fov);
        auto h = tan(theta / 2.0);
        auto viewport_height = 2.0 * h;
        auto viewport_width = aspect_ratio * viewport_height;

        m_w = unit_vector(look_from - look_at);
        m_u = unit_vector(cross(view_up, m_w));
        m_v = cross(m_w, m_u);

        m_origin = look_from;
        m_horizontal = focus_dist * viewport_width * m_u;
        m_vertical = focus_dist * viewport_height * m_v;
        m_left_lower_corner = m_origin - m_horizontal / 2 - m_vertical / 2 - focus_dist * m_w;

        m_lens_radius = aperture / 2;

        m_time0 = time0;
        m_time1 = time1;
    }

    ray get_ray(double s, double t) {
        vec3 rd = m_lens_radius * random_in_unit_disk();
        vec3 offset = m_u * rd.x() + m_v * rd.y();

        return ray{
            m_origin + offset,
            m_left_lower_corner + s * m_horizontal + t * m_vertical - m_origin - offset,
            random_double(m_time0, m_time1)
        };
    }

private:
    point3 m_origin;
    point3 m_left_lower_corner;
    vec3 m_horizontal;
    vec3 m_vertical;
    vec3 m_u, m_v, m_w;
    double m_lens_radius;
    double m_time0, m_time1;
};
