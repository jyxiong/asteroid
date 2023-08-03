#include <glm/detail/type_quat.hpp>
#include "asteroid/renderer/camera_controller.h"

#include "asteroid/input/input.h"
#include "asteroid/util/vec_math.h"
#include "asteroid/util/quaternion.h"

using namespace Asteroid;

CameraController::CameraController()
{
    m_camera.right = cross(m_camera.direction, m_camera.up);
    m_camera.tanHalfFov = tanf(glm::radians(m_camera.verticalFov * 0.5f));
}

bool CameraController::OnUpdate(float ts)
{
    float2 mousePos = Input::GetMousePosition();
    float2 delta = (mousePos - m_camera.lastMousePosition) * 0.002f;
    m_camera.lastMousePosition = mousePos;

    if (!Input::IsMouseButtonDown(MouseButton::Right))
    {
        Input::SetCursorMode(CursorMode::Normal);
        return false;
    }

    Input::SetCursorMode(CursorMode::Locked);

    bool moved = false;

    constexpr float3 upDirection = { 0.0f, 1.0f, 0.0f };
    float3 rightDirection = cross(m_camera.direction, upDirection);

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

        auto q = normalize(Quaternion(-pitchDelta, rightDirection) *
            Quaternion(-yawDelta, { 0.f, 1.0f, 0.0f }));
        auto mat = q.rotationMatrix();

        m_camera.direction = make_float3(mat * make_float4(m_camera.direction, 1.f));
        m_camera.up = make_float3(mat * make_float4(m_camera.up, 1.f));
        m_camera.right = cross(m_camera.direction, m_camera.up);

        moved = true;
    }

    return moved;
}

void CameraController::OnResize(unsigned int width, unsigned int height)
{
    if (width == m_camera.viewport.x && height == m_camera.viewport.y)
        return;

    m_camera.viewport = { width, height };
    m_camera.aspectRatio = (float) m_camera.viewport.x / (float) m_camera.viewport.y;
}
