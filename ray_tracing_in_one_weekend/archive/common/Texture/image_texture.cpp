#include "image_texture.h"
#include <iostream>
#include "Common/External/stb_image.h"
#include "Common/utils.h"

image_texture::image_texture(const char *filename) {

    m_data = stbi_load(filename, &m_width, &m_height, &m_channel, 0);
    if (m_data == nullptr) {
        std::cerr << "ERROR: Could not load texture image file '" << filename << "'.\n";
        m_width = m_height = 0;
    }
}

image_texture::~image_texture() {
    if (m_data != nullptr) {
        stbi_image_free(m_data);
    }
}

color image_texture::value(double u, double v, const point3 &p) const {
    if (m_data == nullptr)
        return color{0.0, 1.0, 1.0};

    u = util::clamp(u, 0.0, 1.0);
    v = 1 - util::clamp(v, 0.0, 1.0); // 翻转y轴，图像空间、纹理空间不一样

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