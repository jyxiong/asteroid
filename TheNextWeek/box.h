#pragma once

#include <memory>
#include "hittable.h"
#include "vec3.h"
#include "hittable_list.h"
#include "material.h"
#include "aarect.h"
#include "ray.h"

class box : public hittable {
public:
    box() = default;
    box(const point3 &box_min, const point3 &box_max, const std::shared_ptr<material>& mat_ptr);

    bool hit(const ray &r, double t_min, double t_max, hit_record &rec) const override;
    bool bounding_box(double time0, double time1, aabb &output_box) const override {
        output_box = aabb(m_box_min, m_box_max);
        return true;
    }

private:
    point3 m_box_min;
    point3 m_box_max;
    hittable_list m_sides;
};

box::box(const point3 &box_min, const point3 &box_max, const std::shared_ptr<material>& mat_ptr)
    : m_box_min(box_min), m_box_max(box_max) {
    m_sides.add(std::make_shared<xy_rect>(m_box_min.x(),
                                          m_box_max.x(),
                                          m_box_min.y(),
                                          m_box_max.y(),
                                          m_box_max.z(),
                                          mat_ptr));
    m_sides.add(std::make_shared<xy_rect>(m_box_min.x(),
                                          m_box_max.x(),
                                          m_box_min.y(),
                                          m_box_max.y(),
                                          m_box_min.z(),
                                          mat_ptr));

    m_sides.add(std::make_shared<xz_rect>(m_box_min.x(),
                                          m_box_max.x(),
                                          m_box_min.z(),
                                          m_box_max.z(),
                                          m_box_max.y(),
                                          mat_ptr));
    m_sides.add(std::make_shared<xz_rect>(m_box_min.x(),
                                          m_box_max.x(),
                                          m_box_min.z(),
                                          m_box_max.z(),
                                          m_box_min.y(),
                                          mat_ptr));

    m_sides.add(std::make_shared<yz_rect>(m_box_min.y(),
                                          m_box_max.y(),
                                          m_box_min.z(),
                                          m_box_max.z(),
                                          m_box_max.x(),
                                          mat_ptr));
    m_sides.add(std::make_shared<yz_rect>(m_box_min.y(),
                                          m_box_max.y(),
                                          m_box_min.z(),
                                          m_box_max.z(),
                                          m_box_min.x(),
                                          mat_ptr));
}

bool box::hit(const ray &r, double t_min, double t_max, hit_record &rec) const {
    return m_sides.hit(r, t_min, t_max, rec);
}