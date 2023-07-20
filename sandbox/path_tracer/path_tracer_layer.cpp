#include "path_tracer_layer.h"

#include "imgui.h"
#include "glm/gtc/type_ptr.hpp"
#include "asteroid/base/application.h"
#include "asteroid/util/timer.h"

using namespace Asteroid;

ExampleLayer::ExampleLayer()
    : Layer("Example") {

    Material &pinkSphere = m_Scene.materials.emplace_back();
    pinkSphere.Albedo = {1.0f, 0.0f, 1.0f};
    pinkSphere.Roughness = 0.0f;

    Material &blueSphere = m_Scene.materials.emplace_back();
    blueSphere.Albedo = {0.2f, 0.3f, 1.0f};
    blueSphere.Roughness = 0.1f;

    {
        Sphere sphere;
        sphere.Position = {0.0f, 0.0f, 0.0f};
        sphere.Radius = 1.0f;
        sphere.MaterialId = 0;
        m_Scene.spheres.push_back(sphere);
    }

    {
        Sphere sphere;
        sphere.Position = {2.0f, .0f, -2.0f};
        sphere.Radius = 0.5f;
        sphere.MaterialId = 1;
        m_Scene.spheres.push_back(sphere);
    }
}

ExampleLayer::~ExampleLayer() = default;

void ExampleLayer::OnAttach() {
}

void ExampleLayer::OnUpdate(float ts) {
    m_CameraController.OnUpdate(ts);
}

void ExampleLayer::OnImGuiRender() {
    ImGui::Begin("Settings");
    ImGui::Text("Last render: %.3fms", m_LastRenderTime);
    if (ImGui::Button("Render")) {
        Render();
    }
    ImGui::End();

    ImGui::Begin("Scene");
    for (size_t i = 0; i < m_Scene.spheres.size(); i++) {
        ImGui::PushID(i);

        Sphere &sphere = m_Scene.spheres[i];
        ImGui::DragFloat3("Position", glm::value_ptr(sphere.Position), 0.1f);
        ImGui::DragFloat("Radius", &sphere.Radius, 0.1f);

        ImGui::DragInt("Material", &sphere.MaterialId, 1.0f, 0, (int) m_Scene.materials.size() - 1);

        ImGui::Separator();

        ImGui::PopID();
    }

    for (size_t i = 0; i < m_Scene.materials.size(); i++) {
        ImGui::PushID(i);

        Material &material = m_Scene.materials[i];
        ImGui::ColorEdit3("Albedo", glm::value_ptr(material.Albedo));
        ImGui::DragFloat("Roughness", &material.Roughness, 0.01f, 0.0f, 1.0f);
        ImGui::DragFloat("Metallic", &material.Metallic, 0.01f, 0.0f, 1.0f);

        ImGui::Separator();

        ImGui::PopID();
    }

    ImGui::End();

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::Begin("Viewport");

    m_ViewportWidth = (unsigned int) ImGui::GetContentRegionAvail().x;
    m_ViewportHeight = (unsigned int) ImGui::GetContentRegionAvail().y;

    auto image = m_Renderer.GetFinalImage();
    if (image)
        ImGui::Image((void *) (intptr_t) image->GetRendererID(),
                     {(float) image->GetWidth(), (float) image->GetHeight()},
                     ImVec2(0, 1), ImVec2(1, 0));

    ImGui::End();
    ImGui::PopStyleVar();

    Render();
}

void ExampleLayer::OnEvent(Event &event) {
}

void ExampleLayer::Render() {
    Timer timer;

    m_Scene.UpdateDevice();

    m_CameraController.OnResize(m_ViewportWidth, m_ViewportHeight);
    m_Renderer.OnResize(m_ViewportWidth, m_ViewportHeight);

    m_Renderer.Render(m_Scene, m_CameraController.GetCamera());

    m_LastRenderTime = timer.ElapsedMillis();
}
