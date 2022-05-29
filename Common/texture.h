#include "vec3.h"
#include "perlin.h"
#include "rtweekend.h"
#include "rtw_stb_image.h"

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
    checker_texture(std::shared_ptr<texture> even, std::shared_ptr<texture> odd)
        : m_even(std::move(even)), m_odd(std::move(odd)) {}
    checker_texture(const color &c1, const color &c2)
        : m_even(std::make_shared<solid_color>(c1)), m_odd(std::make_shared<solid_color>(c2)) {}

    color value(double u, double v, const point3 &p) const override {
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

class noise_texture : public texture {
public:
    noise_texture() = default;
    explicit noise_texture(double scale) : m_scale(scale) {}

    color value(double u, double v, const point3 &p) const override {
//        return color(1, 1, 1) * 0.5 * (1 + m_noise.noise(m_scale * p));
//        return color(1, 1, 1) * m_noise.turb(m_scale * p);
        return color(1, 1, 1) * 0.5 * (1 + sin(m_scale * p.z() + 10 * m_noise.turb(p)));
    }

public:
    perlin m_noise;
    double m_scale{};
};

class image_texture : public texture {
public:
    image_texture() = default;
    explicit image_texture(const char *filename) {

        m_data = stbi_load(filename, &m_width, &m_height, &m_channel, 0);
        if (m_data == nullptr) {
            std::cerr << "ERROR: Could not load texture image file '" << filename << "'.\n";
            m_width = m_height = 0;
        }
    }
    ~image_texture() {
        if (m_data != nullptr) {
            stbi_image_free(m_data);
        }
    }

    color value(double u, double v, const point3 &p) const override {
        if (m_data == nullptr)
            return color{0.0, 1.0, 1.0};

        u = clamp(u, 0.0, 1.0);
        v = 1 - clamp(v, 0.0, 1.0);

        auto i = static_cast<int>(u * m_width);
        auto j = static_cast<int>(v * m_height);

        if (i >= m_width)
            i = m_width - 1;
        if (j >= m_height)
            j = m_height - 1;

        const auto color_scale = 1.0 / 255.0;
        auto pixel = m_data + j * m_channel * m_width + i * m_channel;
        return color{color_scale * pixel[0], color_scale * pixel[1], color_scale * pixel[2]};
    }
private:
    unsigned char *m_data{};
    int m_width{}, m_height{}, m_channel{};
};
