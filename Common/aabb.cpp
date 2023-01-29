#include "Common/aabb.h"

aabb::aabb(const point3 &min, const point3 &max) : m_min(min), m_max(max) {}

aabb aabb::surrounding_box(const aabb &box0, const aabb &box1) {
    point3 small(fmin(box0.min().x(), box1.min().x()),
                 fmin(box0.min().y(), box1.min().y()),
                 fmin(box0.min().z(), box1.min().z()));

    point3 big(fmax(box0.max().x(), box1.max().x()),
               fmax(box0.max().y(), box1.max().y()),
               fmax(box0.max().z(), box1.max().z()));

    return aabb{small, big};
}

bool aabb::hit(const ray &r, double t_min, double t_max) const {
    // 分别计算光线与xyz轴的相交区间，区间的头端点t0取最大值，尾端点t1取最小值
    for (int i = 0; i < 3; i++) {
        auto invD = 1.0 / r.direction()[i];
        auto t0 = (m_min[i] - r.origin()[i]) * invD; // 相交区间的头端点
        auto t1 = (m_max[i] - r.origin()[i]) * invD; // 相交区间的尾端点

        // 如果方向为负数，t0是尾端点，t1是头端点
        if (invD < 0.0)
            std::swap(t0, t1);

        // t_min每次选取最大值，t_max每次选取最小值
        t_min = t0 > t_min ? t0 : t_min;
        t_max = t1 < t_max ? t1 : t_max;

        // 判断最终的区间是否存在
        if (t_max <= t_min)
            return false;
    }
    return true;
}
