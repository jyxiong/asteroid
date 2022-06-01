#pragma once

#include <memory>
#include <utility>
#include "Common/Object/hittable.h"
#include "Common/ray.h"
#include "Common/vec3.h"
#include "Common/randomGenerator.h"
#include "Common/Texture/texture.h"

class material {
public:
    virtual bool scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered) const = 0;
    virtual color emitted(double u , double v, const color &p) const {
        return color{0, 0, 0};
    }
};
