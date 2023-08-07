#include "path_tracer_layer.h"

#include "imgui.h"
#include "glm/gtc/type_ptr.hpp"
#include "asteroid/app/application.h"
#include "asteroid/util/timer.h"

using namespace Asteroid;

ExampleLayer::ExampleLayer()
    : Layer("Example")
{

    Material &pinkSphere = m_Scene.materials.emplace_back();
    pinkSphere.albedo = { 1.0f, 0.0f, 1.0f };
    pinkSphere.roughness = 0.0f;

    Material &blueSphere = m_Scene.materials.emplace_back();
    blueSphere.albedo = { 0.2f, 0.3f, 1.0f };
    blueSphere.roughness = 0.1f;

    Material& orangeSphere = m_Scene.materials.emplace_back();
    orangeSphere.albedo = { 0.8f, 0.5f, 0.2f };
    orangeSphere.roughness = 0.1f;
    orangeSphere.emittance = 2.0f;

    {
        Sphere sphere;
        sphere.position = { 0.0f, 0.0f, 0.0f };
        sphere.radius = 1.0f;
        sphere.materialIndex = 0;
        m_Scene.spheres.push_back(sphere);
    }

    {
        Sphere sphere;
        sphere.position = { 2.0f, 0.0f, 0.0f };
        sphere.radius = 1.0f;
        sphere.materialIndex = 2;
        m_Scene.spheres.push_back(sphere);
    }

    {
        Sphere sphere;
        sphere.position = { 0.0f, -101.0f, 0.0f };
        sphere.radius = 100.0f;
        sphere.materialIndex = 1;
        m_Scene.spheres.push_back(sphere);
    }
}

ExampleLayer::~ExampleLayer() = default;

void ExampleLayer::OnAttach()
{
}

void ExampleLayer::OnUpdate(float ts)
{
    m_modified |= m_CameraController.OnUpdate(ts);
}

void ExampleLayer::OnImGuiRender()
{
    ImGui::Begin("Settings");
    ImGui::Text("Last render: %.3fms", m_LastRenderTime);

    ImGui::Text("Current iteration: %d", m_Renderer.GetRenderState().currentIteration);

    m_modified |= ImGui::DragInt("Trace depth: %d", reinterpret_cast<int *>(&m_Renderer.GetRenderState().traceDepth), 1, 1, 100);

    m_modified |= ImGui::Button("Reset");

    ImGui::End();

    ImGui::Begin("Scene");
    for (size_t i = 0; i < m_Scene.spheres.size(); i++)
    {
        ImGui::PushID(i);

        Sphere &sphere = m_Scene.spheres[i];

        m_modified |= ImGui::DragFloat3("position", glm::value_ptr(sphere.position), 0.1f);
        m_modified |= ImGui::DragFloat("radius", &sphere.radius, 0.1f);
        m_modified |= ImGui::DragInt("material ID", &sphere.materialIndex, 1, 0, (int) m_Scene.materials.size() - 1);

        ImGui::Separator();

        ImGui::PopID();
    }

    ImGui::End();

    ImGui::Begin("Material");

    for (size_t i = 0; i < m_Scene.materials.size(); i++)
    {
        ImGui::PushID(i);

        Material &material = m_Scene.materials[i];

        m_modified |= ImGui::ColorEdit3("albedo", glm::value_ptr(material.albedo));
        m_modified |= ImGui::DragFloat("roughness", &material.roughness, 0.01f, 0.0f, 1.0f);
        m_modified |= ImGui::DragFloat("metallic", &material.metallic, 0.01f, 0.0f, 1.0f);
        m_modified |= ImGui::DragFloat("emittance", &material.emittance, 0.05f, 0.0f, FLT_MAX);

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
                     { (float) image->GetWidth(), (float) image->GetHeight() },
                     ImVec2(0, 1), ImVec2(1, 0));

    ImGui::End();
    ImGui::PopStyleVar();

    Render();
}

void ExampleLayer::Render()
{
    Timer timer;

    if (m_modified)
    {
        m_Renderer.ResetFrameIndex();
        m_modified = false;
    }

    m_Scene.UpdateDevice();

    m_CameraController.OnResize(m_ViewportWidth, m_ViewportHeight);
    m_Renderer.OnResize(m_ViewportWidth, m_ViewportHeight);

    m_Renderer.Render(m_Scene, m_CameraController.GetCamera());

    m_LastRenderTime = timer.ElapsedMillis();
}
