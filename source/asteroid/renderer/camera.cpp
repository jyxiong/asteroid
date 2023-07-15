#include "asteroid/renderer/camera.h"

#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/quaternion.hpp"
#include "glm/gtx/quaternion.hpp"
#include "asteroid/input/input.h"

using namespace Asteroid;

Camera::Camera(float verticalFOV, float nearClip, float farClip)
    : m_VerticalFOV(verticalFOV) {
    m_Position = glm::vec3(0, 0, 1);
    m_Direction = glm::vec3(0, 0, -1);
    m_Up = glm::vec3(0, 1, 0);
    m_Right = glm::cross(m_Direction, m_Up);
    m_focal = glm::vec3(0);
    m_Aspect = (float)m_Viewport.x / (float)m_Viewport.y;
    m_tanHalfFov = tanf(m_VerticalFOV * 0.5f / 180.f * 3.1415926f);
}

void Camera::OnUpdate(float ts)
{
    glm::vec2 mousePos = Input::GetMousePosition();
    glm::vec2 delta = (mousePos - m_LastMousePosition) * 0.002f;
    m_LastMousePosition = mousePos;

    if (!Input::IsMouseButtonDown(MouseButton::Right))
    {
        Input::SetCursorMode(CursorMode::Normal);
        return;
    }

    Input::SetCursorMode(CursorMode::Locked);

    bool moved = false;

    constexpr glm::vec3 upDirection(0.0f, 1.0f, 0.0f);
    glm::vec3 rightDirection = glm::cross(m_Direction, upDirection);

    // Movement
    float moveSpeed = 5.0f;

    if (Input::IsKeyDown(KeyCode::W))
    {
        m_Position += m_Direction * moveSpeed * ts;
        moved = true;
    }
    else if (Input::IsKeyDown(KeyCode::S))
    {
        m_Position -= m_Direction * moveSpeed * ts;
        moved = true;
    }
    if (Input::IsKeyDown(KeyCode::A))
    {
        m_Position -= rightDirection * moveSpeed * ts;
        moved = true;
    }
    else if (Input::IsKeyDown(KeyCode::D))
    {
        m_Position += rightDirection * moveSpeed * ts;
        moved = true;
    }
    if (Input::IsKeyDown(KeyCode::Q))
    {
        m_Position -= upDirection * moveSpeed * ts;
        moved = true;
    }
    else if (Input::IsKeyDown(KeyCode::E))
    {
        m_Position += upDirection * moveSpeed * ts;
        moved = true;
    }

    // Rotation
    if (delta.x != 0.0f || delta.y != 0.0f)
    {
        float rotateSpeed = 0.3f;
        
        float pitchDelta = delta.y * rotateSpeed;
        float yawDelta = delta.x * rotateSpeed;

        glm::quat q = glm::normalize(glm::cross(glm::angleAxis(-pitchDelta, rightDirection),
                                                glm::angleAxis(-yawDelta, glm::vec3(0.f, 1.0f, 0.0f))));
        m_Direction = glm::rotate(q, m_Direction);

        moved = true;
    }
}

void Camera::OnResize(unsigned int width, unsigned int height)
{
    if (width == m_Viewport.x && height == m_Viewport.y)
        return;

    m_Viewport = {width, height};
    m_Aspect = (float)m_Viewport.x / (float)m_Viewport.y;
}
