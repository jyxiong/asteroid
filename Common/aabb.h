#pragma once

#include "vec3.h"
#include "ray.h"

class aabb {
public:
    aabb() = default;

    aabb(const point3 &min, const point3 &max) : m_min(min), m_max(max) {}

    point3 min() const { return m_min; }

    point3 max() const { return m_max; }

    bool hit(const ray &r, double t_min, double t_max) const {
        for (int i = 0; i < 3; i++) {
            auto invD = 1.0 / r.direction()[i];
            auto t0 = (m_min[i] - r.origin()[i]) * invD;
            auto t1 = (m_max[i] - r.origin()[i]) * invD;

            if (invD < 0.0)
                std::swap(t0, t1);

            t_min = t0 > t_min ? t0 : t_min;
            t_max = t1 < t_max ? t1 : t_max;

            if (t_max <= t_min)
                return false;
        }
        return true;
    }

private:
    point3 m_min;
    point3 m_max;
};

aabb surrounding_box(const aabb &box0, const aabb &box1) {
    point3 small(fmin(box0.min().x(), box1.min().x()),
                 fmin(box0.min().y(), box1.min().y()),
                 fmin(box0.min().z(), box1.min().z()));

    point3 big(fmax(box0.max().x(), box1.max().x()),
               fmax(box0.max().y(), box1.max().y()),
               fmax(box0.max().z(), box1.max().z()));

    return {small, big};
}

