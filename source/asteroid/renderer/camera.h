#pragma once

#include <vector>
#include "glm/glm.hpp"
#include "asteroid/renderer/ray.h"

namespace Asteroid
{
struct Camera
{
public:
    Camera(float verticalFOV, float nearClip, float farClip);

    void OnUpdate(float ts);

    void OnResize(unsigned int width, unsigned int height);

private:
    unsigned int m_ViewportWidth = 0, m_ViewportHeight = 0;

    glm::vec3 m_Position;
    glm::vec3 m_View;
    glm::vec3 m_Up;
    glm::vec3 m_Right;
    glm::vec3 m_focal;

    float m_VerticalFOV = 45.0f;
    float m_NearClip = 0.1f;
    float m_FarClip = 100.0f;

    // Cached ray directions
    std::vector<glm::vec3> m_RayDirections;

    glm::vec2 m_LastMousePosition{ 0.0f, 0.0f };
};

}
