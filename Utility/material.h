#pragma once

#include "rtweekend.h"
#include "ray.h"
#include "hittable.h"

class material {
public:
    virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered) const = 0;
};

class lambertian : public material {
public:
    explicit lambertian(const color& albedo) : m_albedo(albedo) {}

    bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered) const override {
        // lambertian material with random scatter direction

        // https://raytracing.github.io/images/fig-1.09-rand-vec.jpg
        // here generate a random point in a unit sphere targeting the sphere at point p
        // reflect light is p->s

        // auto target = rec.p + rec.normal + random_in_unit_sphere();
        // auto target = rec.p + rec.normal + random_unit_vector();
        // auto target = rec.p + random_in_hemisphere(rec.normal);

        vec3 scatter_direction = rec.normal + random_unit_vector();

        if (scatter_direction.near_zero())
            scatter_direction = rec.normal;

        scattered = ray(rec.p, scatter_direction);
        attenuation = m_albedo;

        return true;
    }
private:
    color m_albedo;
};

class metal : public material {
public:
    explicit metal(const color& albedo, double fuzzy) : m_albedo(albedo), m_fuzzy(fuzzy < 1.0 ? fuzzy : 1.0) {}

    bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered) const override {
        // metal material with reflect direction
        vec3 reflect_direction = reflect(r_in.direction(), rec.normal);

        scattered = ray(rec.p, reflect_direction + m_fuzzy * random_in_unit_sphere());
        attenuation = m_albedo;

        return dot(reflect_direction, rec.normal) > 0.0;
    }

private:
    color m_albedo;
    double m_fuzzy;
};