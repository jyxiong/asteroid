#include "path_tracer_layer.h"

#include "imgui.h"
#include "glm/gtc/type_ptr.hpp"
#include "asteroid/app/application.h"
#include "asteroid/util/timer.h"

using namespace Asteroid;

PathTracerLayer::PathTracerLayer() : Layer("Example")
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
    orangeGeometry.emission = glm::vec3(1);

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

    {
        Geometry geometry;
        geometry.type = GeometryType::Cube;
        geometry.translation = { -1.0f, 1.0f, 0.0f };
        geometry.scale = { 0.5f, 0.5f, 0.5f };
        geometry.updateTransform();
        geometry.materialIndex = 1;
        m_Scene.geometries.push_back(geometry);
    }

    m_modified = true;
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

    ImGui::Text("Current frame: %d", m_Renderer.getRenderState().frame);

    m_modified |=
        ImGui::DragInt("Trace depth: %d", reinterpret_cast<int*>(&m_Renderer.getRenderState().maxDepth), 1, 1, 100);

    m_modified |= ImGui::DragInt("Samples per pixel: %d",
                                 reinterpret_cast<int*>(&m_Renderer.getRenderState().maxSamples),
                                 1,
                                 1,
                                 100);

    m_modified |= ImGui::Button("Reset");

    ImGui::End();

    ImGui::Begin("Scene");
    for (size_t i = 0; i < m_Scene.geometries.size(); i++)
    {
        ImGui::PushID(static_cast<int>(i));

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
        ImGui::PushID(static_cast<int>(i));

        Material& material = m_Scene.materials[i];

        m_modified |= ImGui::ColorEdit3("albedo", glm::value_ptr(material.albedo));
        m_modified |= ImGui::DragFloat("roughness", &material.roughness, 0.01f, 0.0f, 1.0f);
        m_modified |= ImGui::DragFloat("metallic", &material.metallic, 0.01f, 0.0f, 1.0f);
        m_modified |= ImGui::DragFloat3("emission", glm::value_ptr(material.emission), 0.05f, 0.0f, FLT_MAX);

        ImGui::Separator();

        ImGui::PopID();
    }

    ImGui::End();

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::Begin("Viewport");

    auto viewport = glm::ivec2(ImGui::GetContentRegionAvail().x, ImGui::GetContentRegionAvail().y);
    if (viewport != m_viewport)
    {
        m_viewport = viewport;
        m_resized = true;
    }

    auto image = m_Renderer.getFinalImage();
    if (image)
        ImGui::Image((void*) (intptr_t) image->rendererID(),
                     { (float) image->width(), (float) image->height() },
                     ImVec2(0, 1),
                     ImVec2(1, 0));

    ImGui::End();
    ImGui::PopStyleVar();

    Render();
}

void PathTracerLayer::Render()
{
    Timer timer;

    if (m_modified)
    {
        m_Scene.UpdateDevice();

        m_Renderer.resetFrameIndex();

        m_modified = false;
    }

    if (m_resized)
    {
        m_CameraController.OnResize(m_viewport);
        m_Renderer.onResize(m_viewport);

        m_resized = false;
    }

    m_Renderer.render(m_Scene, m_CameraController.GetCamera());

    m_LastRenderTime = timer.elapsedMillis();
}
