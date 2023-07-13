#include "asteroid/renderer/camera.h"

#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/quaternion.hpp"
#include "glm/gtx/quaternion.hpp"
#include "asteroid/input/input.h"

using namespace Asteroid;

Camera::Camera(float verticalFOV, float nearClip, float farClip)
    : m_VerticalFOV(verticalFOV), m_NearClip(nearClip), m_FarClip(farClip)
{
    m_View = glm::vec3(0, 0, -1);
    m_Position = glm::vec3(0, 0, 3);
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
    glm::vec3 rightDirection = glm::cross(m_View, upDirection);

    // Movement
    float moveSpeed = 5.0f;

    if (Input::IsKeyDown(KeyCode::W))
    {
        m_Position += m_View * moveSpeed * ts;
        moved = true;
    }
    else if (Input::IsKeyDown(KeyCode::S))
    {
        m_Position -= m_View * moveSpeed * ts;
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
        m_View = glm::rotate(q, m_View);

        moved = true;
    }
}

void Camera::OnResize(uint32_t width, uint32_t height)
{
    if (width == m_ViewportWidth && height == m_ViewportHeight)
        return;

    m_ViewportWidth = width;
    m_ViewportHeight = height;
}
