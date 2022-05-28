#pragma once

#include <vector>
#include <memory>
#include "hittable.h"
#include "Common/ray.h"

class hittable_list : public hittable {
public:
    hittable_list() = default;
    explicit hittable_list(const std::shared_ptr<hittable> &object) { add(object); }

    void clear() { m_objects.clear(); }
    void add(const std::shared_ptr<hittable> &object) { m_objects.push_back(object); }

    bool hit(const ray &r, double t_min, double t_max, hit_record &rec) const override;

private:
    std::vector<std::shared_ptr<hittable>> m_objects;
};

bool hittable_list::hit(const ray &r, double t_min, double t_max, hit_record &rec) const {
    double tmp_t_max = t_max;
    bool hit = false;
    for (const auto &object : m_objects) {
        if (object->hit(r, t_min, tmp_t_max, rec)) {
            tmp_t_max = rec.t;
            hit = true;
        }
    }

    return hit;
}
