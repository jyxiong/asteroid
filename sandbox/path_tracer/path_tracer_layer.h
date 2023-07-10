#pragma once

#include <memory>
#include "glm/glm.hpp"
#include "asteroid/base/layer.h"
#include "asteroid/renderer/image.h"
#include "asteroid/renderer/renderer.h"

namespace Asteroid
{
class ExampleLayer : public Layer
{
public:
    ExampleLayer();

    ~ExampleLayer();

    void OnUpdate() override;

    void OnImGuiRender() override;

    void OnEvent(Event &event) override;

private:
    void Render();

private:
    Renderer m_Renderer;

    unsigned int m_ViewportWidth = 0, m_ViewportHeight = 0;

    float m_LastRenderTime = 0.0f;
};

}