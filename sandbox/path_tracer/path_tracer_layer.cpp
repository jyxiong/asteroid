#include "path_tracer_layer.h"

#include "imgui.h"
#include "glm/gtc/type_ptr.hpp"
#include "asteroid/base/application.h"
#include "asteroid/util/timer.h"

using namespace Asteroid;

ExampleLayer::ExampleLayer()
    : m_Camera(45.0f, 0.1f, 100.0f), Layer("Example")
{
    
    {
        Sphere sphere;
        sphere.Position = { 0.0f, 0.0f, -1.f };
        sphere.Radius = 0.5f;
        sphere.Albedo = { 1.0f, 0.0f, 1.0f };
        m_Scene.spheres.push_back(sphere);
    }

    {
        Sphere sphere;
        sphere.Position = { 1.0f, 0.0f, -5.0f };
        sphere.Radius = 1.5f;
        sphere.Albedo = { 0.2f, 0.3f, 1.0f };
        m_Scene.spheres.push_back(sphere);
    }
	
}

ExampleLayer::~ExampleLayer() = default;

void ExampleLayer::OnAttach()
{
}

void ExampleLayer::OnUpdate(float ts)
{
    m_Camera.OnUpdate(ts);
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

    ImGui::Begin("Scene");
		for (size_t i = 0; i < m_Scene.spheres.size(); i++)
		{
			ImGui::PushID(i);

			Sphere& sphere = m_Scene.spheres[i];
			ImGui::DragFloat3("Position", glm::value_ptr(sphere.Position), 0.1f);
			ImGui::DragFloat("Radius", &sphere.Radius, 0.1f);
			ImGui::ColorEdit3("Albedo", glm::value_ptr(sphere.Albedo));

			ImGui::Separator();

			ImGui::PopID();
		}
    ImGui::End();

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::Begin("Viewport");

    m_ViewportWidth = (unsigned int)ImGui::GetContentRegionAvail().x;
    m_ViewportHeight = (unsigned int)ImGui::GetContentRegionAvail().y;

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

    m_Scene.UpdateDevice();

    m_Camera.OnResize(m_ViewportWidth, m_ViewportHeight);
    m_Renderer.OnResize(m_ViewportWidth, m_ViewportHeight);

    m_Renderer.Render(m_Scene, m_Camera);

    m_LastRenderTime = timer.ElapsedMillis();
}
