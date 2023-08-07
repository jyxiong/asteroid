#include "optix_layer.h"

#include "imgui.h"
#include "glm/gtc/type_ptr.hpp"
#include "asteroid/base/application.h"
#include "asteroid/util/timer.h"

using namespace Asteroid;

OptixLayer::OptixLayer()
    : Layer("OptiX") {

    std::vector<TriangleMesh> model(2);

    model[0].color = glm::vec3(0.f, 1.f, 0.f);
    model[0].addCube(glm::vec3(0.f,-1.5f, 0.f),glm::vec3(10.f,.1f,10.f));
    // a unit cube centered on top of that
    model[1].color = glm::vec3(0.f,1.f,1.f);
    model[1].addCube(glm::vec3(0.f,0.f,0.f),glm::vec3(2.f,2.f,2.f));

    m_renderer.setModel(model);
}

OptixLayer::~OptixLayer() = default;

void OptixLayer::OnAttach() {
}

void OptixLayer::OnUpdate(float ts) {
    m_modified |= m_cameraController.OnUpdate(ts);
}

void OptixLayer::OnImGuiRender() {
    ImGui::Begin("Settings");

    ImGui::Text("Last render: %.3fms", m_LastRenderTime);

    ImGui::End();

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::Begin("Viewport");

    m_ViewportWidth = (unsigned int) ImGui::GetContentRegionAvail().x;
    m_ViewportHeight = (unsigned int) ImGui::GetContentRegionAvail().y;

    auto image = m_renderer.GetFinalImage();
    if (image)
        ImGui::Image((void *) (intptr_t) image->GetRendererID(),
                     {(float) image->GetWidth(), (float) image->GetHeight()},
                     ImVec2(0, 1), ImVec2(1, 0));

    ImGui::End();
    ImGui::PopStyleVar();

    Render();
}

void OptixLayer::Render() {
    Timer timer;

    m_cameraController.OnResize(m_ViewportWidth, m_ViewportHeight);

    m_renderer.OnResize(m_ViewportWidth, m_ViewportHeight);

    m_renderer.setCamera(m_cameraController.GetCamera());
    m_renderer.Render();

    m_LastRenderTime = timer.ElapsedMillis();
}
