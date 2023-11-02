#pragma once

#include <memory>
#include "asteroid/renderer/scene_struct.h"

namespace Asteroid
{

class CameraController
{

public:
    CameraController();

    Camera& GetCamera() { return m_camera; }

    const Camera& GetCamera() const { return m_camera; }

    bool OnUpdate(float ts);

    void OnResize(const glm::ivec2& viewport);

private:
    Camera m_camera;
};

}