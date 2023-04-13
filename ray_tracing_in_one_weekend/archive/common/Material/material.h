#pragma once

#include <memory>
#include <utility>
#include "Common/Object/hittable.h"
#include "Common/Texture/texture.h"
#include "Common/ray.h"
#include "Common/rng.h"
#include "Common/vec3.h"

class material {
public:
    // 散射函数，根据入射光线和碰撞点信息，计算散射光线和颜色
    virtual bool scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered) const = 0;
    virtual color emitted(double u , double v, const color &p) const {
        return color{0, 0, 0};
    }
};
