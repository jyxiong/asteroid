#pragma once

#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

void write_color(const std::vector<glm::u8vec3> &framebuffer, unsigned int width, unsigned int height)
{
    stbi_flip_vertically_on_write(true);
    stbi_write_jpg("one_weekend.jpg", (int) width, (int) height, 3, framebuffer.data(), 100);
}