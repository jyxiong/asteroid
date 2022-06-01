#pragma once

#include <memory>
#include "Common/Object/hittable.h"
#include "Common/Object/hittable_list.h"
#include "Common/Material/material.h"
#include "Common/ray.h"
#include "Common/vec3.h"

class box : public hittable {
public:
    box() = default;
    box(const point3 &box_min, const point3 &box_max, const std::shared_ptr<material>& mat_ptr);

    bool hit(const ray &r, double t_min, double t_max, hit_record &rec) const override;
    bool bounding_box(double time0, double time1, aabb &output_box) const override;

private:
    point3 m_box_min;
    point3 m_box_max;
    hittable_list m_sides;
};
