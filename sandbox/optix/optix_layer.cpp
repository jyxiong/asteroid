#include "optix_layer.h"

#include "imgui.h"
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
    ImGui::Begin("Settings");

    ImGui::Text("Last render: %.3fms", m_LastRenderTime);

    ImGui::End();

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::Begin("Viewport");

    m_ViewportWidth = (unsigned int) ImGui::GetContentRegionAvail().x;
    m_ViewportHeight = (unsigned int) ImGui::GetContentRegionAvail().y;

    auto image = m_Renderer.GetFinalImage();
    if (image)
        ImGui::Image((void *) (intptr_t) image->GetRendererID(),
                     { (float) image->GetWidth(), (float) image->GetHeight() },
                     ImVec2(0, 1), ImVec2(1, 0));

    ImGui::End();
    ImGui::PopStyleVar();

    Render();
}

void OptixLayer::OnEvent(Event &event)
{
}

void OptixLayer::Render()
{
    Timer timer;

    m_Renderer.OnResize(m_ViewportWidth, m_ViewportHeight);

    m_Renderer.Render();

    m_LastRenderTime = timer.ElapsedMillis();
}
