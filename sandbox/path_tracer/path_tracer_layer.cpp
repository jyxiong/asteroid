#include "path_tracer_layer.h"

#include "glad/gl.h"
#include "cuda_gl_interop.h"
#include "imgui.h"
#include "glm/glm.hpp"

#include "asteroid/base/application.h"
#include "asteroid/util/timer.h"

using namespace Asteroid;

ExampleLayer::ExampleLayer()
    : Layer("Example")
{
}

ExampleLayer::~ExampleLayer()
{
}

void ExampleLayer::OnUpdate()
{
    
}

void ExampleLayer::OnImGuiRender()
{
    ImGui::Begin("Settings");
    ImGui::Text("Last render: %.3fms", m_LastRenderTime);
    if (ImGui::Button("Render"))
    {
        Render();
    }
    ImGui::End();

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::Begin("Viewport");

    m_ViewportWidth = ImGui::GetContentRegionAvail().x;
    m_ViewportHeight = ImGui::GetContentRegionAvail().y;

    auto image = m_Renderer.GetFinalImage();
    if (image)
        ImGui::Image((void*)(intptr_t)image->GetRendererID(), { (float)image->GetWidth(), (float)image->GetHeight() },
            ImVec2(0, 1), ImVec2(1, 0));

    ImGui::End();
    ImGui::PopStyleVar();

    Render();
}

void ExampleLayer::OnEvent(Event &event)
{
}

void ExampleLayer::Render()
{
    Timer timer;
    
    m_Renderer.OnResize(m_ViewportWidth, m_ViewportHeight);
    m_Renderer.Render();

    m_LastRenderTime = timer.ElapsedMillis();
}
