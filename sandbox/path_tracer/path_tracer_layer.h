#pragma once

#include <memory>
#include "glm/glm.hpp"
#include "asteroid/app/layer.h"
#include "asteroid/app/image.h"
#include "asteroid/renderer/scene.h"
#include "asteroid/renderer/camera_controller.h"
#include "renderer.h"

namespace Asteroid
{
class PathTracerLayer : public Layer
{
public:
    PathTracerLayer();

    ~PathTracerLayer() override;

    void OnAttach() override;

    void OnUpdate(float ts) override;

    void OnImGuiRender() override;

private:
    void Render();

private:
    Renderer m_Renderer;

    CameraController m_CameraController;

    Scene m_Scene;

    unsigned int m_ViewportWidth = 0, m_ViewportHeight = 0;

    bool m_modified = false;

    float m_LastRenderTime = 0.0f;
};

}