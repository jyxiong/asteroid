#include "asteroid/app/application.h"

#include <memory>
#include "asteroid/util/log.h"
#include "asteroid/util/macro.h"

using namespace Asteroid;

Application *Application::s_Instance = nullptr;

Application::Application()
{
    AST_ASSERT(!s_Instance, "Application already exists!")
    s_Instance = this;

    InitWindow();

    InitLayer();
}

Application::~Application() = default;

void Application::PushLayer(const std::shared_ptr<Layer> &layer)
{
    m_LayerStack.emplace_back(layer);
    layer->OnAttach();
}

void Application::Run()
{
    while (!m_Window->ShouldClose())
    {
        auto time = (float) glfwGetTime();
        m_FrameTime = time - m_LastFrameTime;
        m_TimeStep = min(m_FrameTime, 0.0333f);
        m_LastFrameTime = time;

        // update
        for (auto layer : m_LayerStack)
            layer->OnUpdate(m_TimeStep);

        // imgui
        m_ImGuiLayer->Begin();

        for (auto layer : m_LayerStack)
            layer->OnImGuiRender();

        m_ImGuiLayer->End();

        // glfw
        m_Window->OnUpdate();
    }
}

void Application::InitWindow()
{
    m_Window = std::unique_ptr<Window>(Window::Create());
}

void Application::InitLayer()
{
    m_ImGuiLayer = std::make_shared<ImGuiLayer>();
    PushLayer(m_ImGuiLayer);
}
