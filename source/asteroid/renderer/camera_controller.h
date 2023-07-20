#pragma once

#include <memory>
#include "asteroid/renderer/camera.h"

namespace Asteroid {

class CameraController {

public:
    CameraController();

    Camera& GetCamera() { return m_camera; }

    const Camera& GetCamera() const { return m_camera; }

    void OnUpdate(float ts);

    void OnResize(unsigned int width, unsigned int height);

private:
    Camera m_camera;
};

}