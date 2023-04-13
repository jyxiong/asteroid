#pragma once

#include "vec3.h"
#include "ray.h"

class aabb {
public:
    aabb() = default;
    aabb(const point3 &min, const point3 &max);
    static aabb surrounding_box(const aabb &box0, const aabb &box1);

    point3 min() const { return m_min; }
    point3 max() const { return m_max; }

    bool hit(const ray &r, double t_min, double t_max) const;
private:
    point3 m_min;
    point3 m_max;
};
