#pragma once

#include <memory>
#include "glm/glm.hpp"
#include "asteroid/base/layer.h"
#include "asteroid/base/image.h"
#include "asteroid/renderer/renderer.h"
#include "asteroid/renderer/scene.h"
#include "asteroid/renderer/camera_controller.h"

namespace Asteroid
{
class ExampleLayer : public Layer
{
public:
    ExampleLayer();

    ~ExampleLayer();

    void OnAttach() override;

    void OnUpdate(float ts) override;

    void OnImGuiRender() override;

    void OnEvent(Event &event) override;

private:
    void Render();

private:
    Renderer m_Renderer;

    CameraController m_CameraController;

    Scene m_Scene;

    unsigned int m_ViewportWidth = 0, m_ViewportHeight = 0;

    float m_LastRenderTime = 0.0f;
};

}