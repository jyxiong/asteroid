#pragma once

#include <memory>
#include "glm/glm.hpp"
#include "asteroid/app/layer.h"
#include "asteroid/app/image.h"
#include "asteroid/app/camera_controller.h"
#include "asteroid/renderer/scene.h"
#include "asteroid/renderer/renderer.h"

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

    glm::ivec2 m_viewport;

    bool m_modified = false;
    bool m_resized = false;

    float m_LastRenderTime = 0.0f;
};

}