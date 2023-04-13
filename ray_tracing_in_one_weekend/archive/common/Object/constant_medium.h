#pragma once

#include <memory>
#include <iostream>

#include "Common/Object/hittable.h"
#include "Common/Material/material.h"
#include "Common/Material/isotropic.h"
#include "Common/Texture/texture.h"

class constant_medium : public hittable {
public:
    constant_medium(std::shared_ptr<hittable> object_ptr, double density, std::shared_ptr<texture> albedo);

    constant_medium(std::shared_ptr<hittable> object_ptr, double density, color albedo);

    bool hit(const ray &r, double t_min, double t_max, hit_record &rec) const override;
    bool bounding_box(double time0, double time1, aabb &output_box) const override {
        return m_boundary->bounding_box(time0, time1, output_box);
    }

public:
    std::shared_ptr<hittable> m_boundary;
    std::shared_ptr<material> m_phase_function;
    double m_neg_inv_density;
};

