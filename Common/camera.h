#pragma once

#include "vec3.h"
#include "ray.h"

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
    );

    ray get_ray(double s, double t);

private:
    point3 m_origin;
    point3 m_left_lower_corner;
    vec3 m_horizontal;
    vec3 m_vertical;
    vec3 m_u, m_v, m_w;
    double m_lens_radius;
    double m_time0, m_time1; // 快门的开关时间，用于控制光线的时间
};
