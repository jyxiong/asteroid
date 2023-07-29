#include "optix_layer.h"

#include "imgui.h"
#include "glm/gtc/type_ptr.hpp"
#include "asteroid/base/application.h"
#include "asteroid/util/timer.h"

using namespace Asteroid;

OptixLayer::OptixLayer()
    : Layer("OptiX") {}

OptixLayer::~OptixLayer() = default;

void OptixLayer::OnAttach()
{
}

void OptixLayer::OnUpdate(float ts)
{

}

void OptixLayer::OnImGuiRender()
{
    Render();
}

void OptixLayer::OnEvent(Event &event)
{
}

void OptixLayer::Render()
{
    m_Renderer.OnResize(m_ViewportWidth, m_ViewportHeight);

    m_Renderer.Render();
}
