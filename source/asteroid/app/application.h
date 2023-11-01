#pragma once

#include "asteroid/app/window.h"
#include "asteroid/app/layer.h"
#include "asteroid/app/imgui/imgui_layer.h"

namespace Asteroid
{

class Application
{
public:
    Application();

    virtual ~Application();

    inline Window& GetWindow() { return *m_Window; }

    void PushLayer(const std::shared_ptr<Layer>& layer);

    void Run();

public:
    inline static Application& Get() { return *s_Instance; }

private:

    void InitWindow();

    void InitLayer();

private:

    std::unique_ptr<Window> m_Window;

    std::shared_ptr<ImGuiLayer> m_ImGuiLayer;

    std::vector<std::shared_ptr<Layer>> m_LayerStack;

    float m_TimeStep = 0.0f;
    float m_FrameTime = 0.0f;
    float m_LastFrameTime = 0.0f;

    static Application* s_Instance;
};

// 该函数由派生的应用定义
Application* CreateApplication();

}
