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

class dielectric : public material {
public:
    explicit dielectric(double ir) : m_ir(ir) {}

    bool scatter(const ray& r, const hit_record& rec, color& attenuation, ray& scattered) const override {
        // front_face: from air(1.0) to dielectric(m_ir)
        double refraction_ratio = rec.front_face ? (1.0 / m_ir) : m_ir;

        auto in_direction = unit_vector(r.direction());
        auto cos_theta = fmin(dot(-in_direction, rec.normal), 1.0);
        auto sin_theta = sqrt(1 - cos_theta * cos_theta);

        vec3 scatter_direction;
        bool cannot_refract = refraction_ratio * sin_theta > 1.0;
        if (cannot_refract || reflectance(cos_theta, refraction_ratio) > random_double()) // maybe not refraction
            scatter_direction = reflect(in_direction, rec.normal);
        else
            scatter_direction = refract(in_direction, rec.normal, refraction_ratio);

        scattered = ray(rec.p, scatter_direction);
        attenuation = color { 1.0, 1.0, 1.0 };

        return true;
    }

private:
    static double reflectance(double cosine, double ir) {
        // Schlick approximation
        auto r0 = (1 - ir) / (1.0 + ir);
        r0 = r0 * r0;
        return r0 + (1 - r0) * pow(1 - cosine, 5);
    }

private:
    double m_ir; // index of refraction
};