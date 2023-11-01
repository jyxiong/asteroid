#include "path_tracer_layer.h"

#include "imgui.h"
#include "glm/gtc/type_ptr.hpp"
#include "asteroid/app/application.h"
#include "asteroid/util/timer.h"

using namespace Asteroid;

PathTracerLayer::PathTracerLayer()
    : Layer("Example")
{

    Material& pinkGeometry = m_Scene.materials.emplace_back();
    pinkGeometry.albedo = { 1.0f, 0.0f, 1.0f };
    pinkGeometry.roughness = 0.0f;

    Material& blueGeometry = m_Scene.materials.emplace_back();
    blueGeometry.albedo = { 0.2f, 0.3f, 1.0f };
    blueGeometry.roughness = 0.1f;

    Material& orangeGeometry = m_Scene.materials.emplace_back();
    orangeGeometry.albedo = { 0.8f, 0.5f, 0.2f };
    orangeGeometry.roughness = 0.1f;
    orangeGeometry.emittance = 2.0f;

    {
        Geometry geometry;
        geometry.type = GeometryType::Sphere;
        geometry.materialIndex = 0;
        geometry.updateTransform();
        m_Scene.geometries.push_back(geometry);
    }

    {
        Geometry geometry;
        geometry.type = GeometryType::Sphere;
        geometry.translation = { 2.0f, 0.0f, 0.0f };
        geometry.updateTransform();
        geometry.materialIndex = 2;
        m_Scene.geometries.push_back(geometry);
    }

    {
        Geometry geometry;
        geometry.type = GeometryType::Sphere;
        geometry.translation = { 0.0f, -101.0f, 0.0f };
        geometry.scale = { 100.0f, 100.0f, 100.0f };
        geometry.updateTransform();
        geometry.materialIndex = 1;
        m_Scene.geometries.push_back(geometry);
    }
}

PathTracerLayer::~PathTracerLayer() = default;

void PathTracerLayer::OnAttach()
{
}

void PathTracerLayer::OnUpdate(float ts)
{
    m_modified |= m_CameraController.OnUpdate(ts);
}

void PathTracerLayer::OnImGuiRender()
{
    ImGui::Begin("Settings");
    ImGui::Text("Last render: %.3fms", m_LastRenderTime);

    ImGui::Text("Current iteration: %d", m_Renderer.GetRenderState().currentIteration);

    m_modified |=
        ImGui::DragInt("Trace depth: %d", reinterpret_cast<int*>(&m_Renderer.GetRenderState().traceDepth), 1, 1, 100);

    m_modified |= ImGui::Button("Reset");

    ImGui::End();

    ImGui::Begin("Scene");
    for (size_t i = 0; i < m_Scene.geometries.size(); i++)
    {
        ImGui::PushID(i);

        Geometry& geometry = m_Scene.geometries[i];

        m_modified |= ImGui::DragFloat3("translation", glm::value_ptr(geometry.translation), 0.1f);
        m_modified |= ImGui::DragFloat3("rotation", glm::value_ptr(geometry.rotation), 0.1f);
        m_modified |= ImGui::DragFloat3("scale", glm::value_ptr(geometry.scale), 0.1f);
        m_modified |= ImGui::DragInt("material ID", &geometry.materialIndex, 1, 0, (int) m_Scene.materials.size() - 1);

        ImGui::Separator();

        ImGui::PopID();
    }

    ImGui::End();

    ImGui::Begin("Material");

    for (size_t i = 0; i < m_Scene.materials.size(); i++)
    {
        ImGui::PushID(i);

        Material& material = m_Scene.materials[i];

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

    m_ViewportWidth = (int) ImGui::GetContentRegionAvail().x;
    m_ViewportHeight = (int) ImGui::GetContentRegionAvail().y;

    auto image = m_Renderer.GetFinalImage();
    if (image)
        ImGui::Image((void*) (intptr_t) image->GetRendererID(),
                     { (float) image->GetWidth(), (float) image->GetHeight() },
                     ImVec2(0, 1), ImVec2(1, 0));

    ImGui::End();
    ImGui::PopStyleVar();

    Render();
}

void PathTracerLayer::Render()
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
