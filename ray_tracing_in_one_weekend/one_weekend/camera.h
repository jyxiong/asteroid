#pragma once

#include "glm/glm.hpp"
#include "glm/ext/scalar_constants.hpp"

#include "ray.h"

/**
 *
 * NDC：范围是[-1, 1]^3
 * 屏幕坐标系：范围是[0, width]和[0, height]
 */

struct CameraParameters
{
    float fov;
    float focal_length;
    glm::uvec2 film_size;
};

class Camera
{
public:
    glm::vec3 position;
    glm::vec3 direction;
    glm::vec3 up;
    glm::vec3 right;

    float fov{};
    float focal_length{};
    glm::uvec2 film_size;

    float aspect{};
    float tan_half_fov{};

public:
    Camera() = default;

    Camera(const glm::vec3 &position, const glm::vec3 &direction, const glm::vec3 &up)
        : position(position), direction(direction), up(up)
    {
        right = glm::cross(direction, up);
    }

    void set_parameters(const CameraParameters &params)
    {
        fov = params.fov;
        focal_length = params.focal_length;
        film_size = params.film_size;

        aspect = (float) film_size.x / (float) film_size.y;
        tan_half_fov = tanf(glm::pi<float>() * fov / 180.f);
    }

    __device__ Ray generate_ray(unsigned int x, unsigned int y) const
    {
        // 焦平面转换到NDC
        auto nx = 2.f * (((float) x + 0.5f) / (float) film_size.x) - 1.f;
        auto ny = 1.f - 2.f * (((float) y + 0.5f) / (float) film_size.y);

        auto offset_x = nx * tan_half_fov * aspect;
        auto offset_y = ny * tan_half_fov;

        return { position, glm::normalize(direction + right * offset_x + up * offset_y) };
    }
};
