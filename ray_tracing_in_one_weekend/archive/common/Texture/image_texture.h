#pragma once

#include "Common/vec3.h"
#include "texture.h"

class image_texture : public texture {
public:
    image_texture() = default;
    explicit image_texture(const char *filename);
    ~image_texture();

    color value(double u, double v, const point3 &p) const override;
private:
    unsigned char *m_data{};
    int m_width{}, m_height{}, m_channel{};
};