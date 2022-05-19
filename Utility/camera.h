#pragma once

#include "vec3.h"
#include "ray.h"

class camera {
public:
    camera() {
        const double aspect_ratio = 16.0 / 9.0;
        auto focal_length = 1.0;
        auto viewport_height = 2.0;
        auto viewport_width = viewport_height * aspect_ratio;

        m_origin = vec3(0.0, 0.0, 0.0);
        m_horizontal = vec3(viewport_width, 0.0, 0.0);
        m_vertical = vec3(0.0, viewport_height, 0.0);
        m_left_lower_corner = m_origin - m_horizontal / 2 - m_vertical / 2 - vec3(0.0, 0.0, focal_length);
    }

    ray get_ray(double u, double v) {
        return ray{m_origin, m_left_lower_corner + u * m_horizontal + v * m_vertical - m_origin};
    }

private:
    point3 m_origin;
    vec3 m_horizontal;
    vec3 m_vertical;
    point3 m_left_lower_corner;
};
