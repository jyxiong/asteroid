#pragma once

#include <memory>
#include "glm/glm.hpp"
#include "asteroid/base/layer.h"
#include "asteroid/base/image.h"
#include "asteroid/renderer/scene.h"
#include "asteroid/renderer/camera_controller.h"
#include "renderer.h"

namespace Asteroid {
class OptixLayer : public Layer {
public:
    OptixLayer();

    ~OptixLayer() override;

    void OnAttach() override;

    void OnUpdate(float ts) override;

    void OnImGuiRender() override;

private:
    void Render();

private:
    Renderer m_renderer;

    CameraController m_cameraController;

    unsigned int m_ViewportWidth = 0, m_ViewportHeight = 0;

    bool m_modified = false;

    float m_LastRenderTime = 0.0f;
};

}