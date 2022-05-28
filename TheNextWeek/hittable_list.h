#pragma once

#include <vector>
#include <memory>
#include "hittable.h"
#include "Common/ray.h"

class hittable_list : public hittable {
public:
    hittable_list() = default;
    explicit hittable_list(const std::shared_ptr<hittable>& object) { add(object); }

    void clear() { objects.clear(); }
    void add(const std::shared_ptr<hittable>& object) { objects.push_back(object); }

    bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const override;
    bool bounding_box(double time0, double time1, aabb &output_box) const override;

public:
    std::vector<std::shared_ptr<hittable>> objects;
};

bool hittable_list::hit(const ray& r, double t_min, double t_max, hit_record& rec) const {
    double tmp_t_max = t_max;
    bool hit = false;
    for (const auto& object : objects) {
        if (object->hit(r, t_min, tmp_t_max, rec)) {
            tmp_t_max = rec.t;
            hit = true;
        }
    }

    return hit;
}

bool hittable_list::bounding_box(double time0, double time1, aabb &output_box) const {
    if (objects.empty())
        return false;

    aabb temp_box;
    bool first_box = true;

    for (const auto &object : objects) {
        if (!object->bounding_box(time0, time1, temp_box))
            return false;
        output_box = first_box ? temp_box : surrounding_box(output_box, temp_box);
        first_box = false;
    }
    return true;
}