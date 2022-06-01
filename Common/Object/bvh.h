#pragma once

#include <memory>
#include <vector>
#include <iostream>
#include <algorithm>
#include "Common/Object/hittable.h"
#include "Common/Object/hittable_list.h"
#include "Common/randomGenerator.h"

class bvh_node : public hittable {
public:
    bvh_node() = default;
    bvh_node(const std::vector<std::shared_ptr<hittable>> &src_objects,
             size_t start,
             size_t end,
             double time0,
             double time1);
    bvh_node(const hittable_list &list, double time0, double time1);

    bool hit(const ray &r, double t_min, double t_max, hit_record &rec) const override;
    bool bounding_box(double time0, double time1, aabb &output_box) const override;

private:
    std::shared_ptr<hittable> m_left;
    std::shared_ptr<hittable> m_right;
    aabb m_box;
};

inline bool box_compare(const std::shared_ptr<hittable> &a, const std::shared_ptr<hittable> &b, int axis);

bool box_x_compare(const std::shared_ptr<hittable> &a, const std::shared_ptr<hittable> &b);

bool box_y_compare(const std::shared_ptr<hittable> &a, const std::shared_ptr<hittable> &b);

bool box_z_compare(const std::shared_ptr<hittable> &a, const std::shared_ptr<hittable> &b);
