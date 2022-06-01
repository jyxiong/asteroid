#pragma once

#include <memory>
#include <utility>
#include "Common/Object/hittable.h"

class moving_sphere : public hittable {
public:
    moving_sphere() = default;
    moving_sphere(const vec3 &center0,
                  const vec3 &center1,
                  double time0,
                  double time1,
                  double radius,
                  std::shared_ptr<material> mat_ptr);

    bool hit(const ray &r, double t_min, double t_max, hit_record &rec) const override;
    bool bounding_box(double time0, double time1, aabb &output_box) const override;
private:
    vec3 center(double time) const;

private:
    vec3 m_center0, m_center1;
    double m_time0{}, m_time1{};
    double m_radius{};
    std::shared_ptr<material> m_mat_ptr;
};
