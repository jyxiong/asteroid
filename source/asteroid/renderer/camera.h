#pragma once

#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include "asteroid/renderer/ray.h"

namespace Asteroid
{
class Camera
{
public:
    Camera(float verticalFOV, float nearClip, float farClip);

    void OnUpdate(float ts);

    void OnResize(unsigned int width, unsigned int height);

    __device__ const glm::uvec2 &GetViewport() const { return m_Viewport; }

    __device__ void GeneratePrimaryRay(const glm::vec2 &uv, Ray &ray) const
    {
        auto x = float(uv.x) * m_tanHalfFov * m_Aspect * m_Right;
        auto y = float(uv.y) * m_tanHalfFov * m_Up;
        ray.Direction = glm::normalize(m_Direction + x + y);
        ray.Origin = m_Position;
    }

private:

    glm::uvec2 m_Viewport{};

    glm::vec3 m_Position{};
    glm::vec3 m_Direction{};
    glm::vec3 m_Up{};
    glm::vec3 m_Right{};
    glm::vec3 m_focal{};

    float m_VerticalFOV;
    float m_tanHalfFov;
    float m_Aspect;

    glm::vec2 m_LastMousePosition{ 0.0f, 0.0f };
};

}
