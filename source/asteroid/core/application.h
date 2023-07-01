#pragma once

#include "asteroid/event/event.h"
#include "asteroid/event/application_event.h"
#include "asteroid/core/window.h"
#include "asteroid/core/layer.h"
#include "asteroid/core/layer_stack.h"

namespace Asteroid {

class Application
{
public:
    Application();
    virtual ~Application();

    inline Window& GetWindow() { return *m_Window; }

    void PushLayer(Layer* layer);
    void PushOverlay(Layer* layer);

    void OnEvent(Event& e);

    void Run();

public:
    inline static Application& Get() { return *s_Instance; }

private:

    bool OnWindowClose(WindowCloseEvent& e);

    std::unique_ptr<Window> m_Window;
    bool m_Running = true;

    LayerStack m_LayerStack;

private:
    static Application* s_Instance;
};

// 该函数由派生的应用定义
Application* CreateApplication();

}
