#pragma once

#include <memory>
#include <utility>
#include "Common/Object/hittable.h"
#include "Common/ray.h"
#include "Common/vec3.h"
#include "Common/aabb.h"
#include "Common/utils.h"

class sphere : public hittable {
public:
    sphere() = default;
    sphere(const point3 &center, double radius);
    sphere(const point3 &center, double radius, std::shared_ptr<material> mat_ptr);

    bool hit(const ray &r, double t_min, double t_max, hit_record &rec) const override;
    bool bounding_box(double time0, double time1, aabb &output_box) const override;

private:
    point3 m_center;
    double m_radius{};
    std::shared_ptr<material> m_mat_ptr;
};


