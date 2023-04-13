#pragma once

#include <vector>
#include <memory>
#include "Common/Object/hittable.h"
#include "Common/ray.h"

class hittable_list : public hittable {
public:
    hittable_list() = default;
    explicit hittable_list(const std::shared_ptr<hittable>& object);

    void clear();
    void add(const std::shared_ptr<hittable>& object);

    bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const override;
    bool bounding_box(double time0, double time1, aabb &output_box) const override;

public:
    std::vector<std::shared_ptr<hittable>> objects;
};
