#pragma once

#include "asteroid/event/event.h"
#include "asteroid/event/application_event.h"
#include "asteroid/base/window.h"
#include "asteroid/base/layer.h"
#include "asteroid/base/layer_stack.h"
#include "asteroid/imgui/imgui_layer.h"

namespace Asteroid {

class Application
{
public:
    Application();
    virtual ~Application();

    inline Window& GetWindow() { return *m_Window; }

    void PushLayer(Layer* layer);
    void PushOverlay(Layer* layer);

    void Run();

public:
    inline static Application& Get() { return *s_Instance; }

private:

    void InitWindow();

    void InitLayer();

    void OnEvent(Event& e);

    bool OnWindowClose(WindowCloseEvent& e);

    bool OnWindowResize(WindowResizeEvent& e);

    std::unique_ptr<Window> m_Window;
    ImGuiLayer* m_ImGuiLayer;
    bool m_Running = true;
    bool m_Minimized = false;
    LayerStack m_LayerStack;

private:
    static Application* s_Instance;
};

// 该函数由派生的应用定义
Application* CreateApplication();

}
