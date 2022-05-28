#include "vec3.h"
#include "rtweekend.h"

class texture {
public:
    virtual color value(double u, double v, const point3 &p) const = 0;
};

class solid_color : public texture {
public:
    solid_color() = default;
    explicit solid_color(const color &c) : m_color(c) {}

    solid_color(double r, double g, double b) : solid_color(color{r, g, b}) {}

    color value(double u, double v, const vec3 &p) const override {
        return m_color;
    }

private:
    color m_color;
};

class checker_texture : public texture {
public:
    checker_texture() = default;
    checker_texture(const std::shared_ptr<texture> &even, const std::shared_ptr<texture> &odd)
        : m_even(even), m_odd(odd) {}
    checker_texture(const color &c1, const color &c2)
        : m_even(std::make_shared<solid_color>(c1)), m_odd(std::make_shared<solid_color>(c2)) {}

    virtual color value(double u, double v, const point3 &p) const override {
        auto sines = sin(10 * p.x()) * sin(10 * p.y()) * sin(10 * p.z());
        if (sines < 0)
            return m_odd->value(u, v, p);
        else
            return m_even->value(u, v, p);
    }
private:
    std::shared_ptr<texture> m_even;
    std::shared_ptr<texture> m_odd;
};