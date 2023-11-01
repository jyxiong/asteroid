#include "asteroid/renderer/camera_controller.h"

#include "glm/gtc/quaternion.hpp"
#include "glm/gtx/quaternion.hpp"
#include "asteroid/app/input/input.h"

using namespace Asteroid;

CameraController::CameraController()
{
    m_camera.right = glm::cross(m_camera.direction, m_camera.up);
    m_camera.tanHalfFov = glm::tan(glm::radians(m_camera.verticalFov * 0.5f));
}

bool CameraController::OnUpdate(float ts)
{
    glm::vec2 mousePos = Input::GetMousePosition();
    glm::vec2 delta = (mousePos - m_camera.lastMousePosition) * 0.002f;
    m_camera.lastMousePosition = mousePos;

    if (!Input::IsMouseButtonDown(MouseButton::Right))
    {
        Input::SetCursorMode(CursorMode::Normal);
        return false;
    }

    Input::SetCursorMode(CursorMode::Locked);

    bool moved = false;

    constexpr glm::vec3 upDirection(0.0f, 1.0f, 0.0f);
    glm::vec3 rightDirection = glm::cross(m_camera.direction, upDirection);

    // Movement
    float moveSpeed = 5.0f;

    if (Input::IsKeyDown(KeyCode::W))
    {
        m_camera.position += m_camera.direction * moveSpeed * ts;
        moved = true;
    } else if (Input::IsKeyDown(KeyCode::S))
    {
        m_camera.position -= m_camera.direction * moveSpeed * ts;
        moved = true;
    }
    if (Input::IsKeyDown(KeyCode::A))
    {
        m_camera.position -= rightDirection * moveSpeed * ts;
        moved = true;
    } else if (Input::IsKeyDown(KeyCode::D))
    {
        m_camera.position += rightDirection * moveSpeed * ts;
        moved = true;
    }
    if (Input::IsKeyDown(KeyCode::Q))
    {
        m_camera.position -= upDirection * moveSpeed * ts;
        moved = true;
    } else if (Input::IsKeyDown(KeyCode::E))
    {
        m_camera.position += upDirection * moveSpeed * ts;
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

        m_camera.direction = glm::rotate(q, m_camera.direction);
        m_camera.up = glm::rotate(q, m_camera.up);
        m_camera.right = glm::cross(m_camera.direction, m_camera.up);

        moved = true;
    }

    return moved;
}

void CameraController::OnResize(int width, int height)
{
    if (width == m_camera.viewport.x && height == m_camera.viewport.y)
        return;

    m_camera.viewport = { width, height };
    m_camera.aspectRatio = (float) m_camera.viewport.x / (float) m_camera.viewport.y;
}
