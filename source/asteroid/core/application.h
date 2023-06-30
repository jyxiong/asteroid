#pragma once

#include "asteroid/event/event.h"
#include "asteroid/event/application_event.h"
#include "asteroid/core/window.h"

namespace Asteroid {

class Application
{
public:
    Application();
    virtual ~Application();

    void Run();

    void OnEvent(Event& e);
private:
    bool OnWindowClose(WindowCloseEvent& e);

    std::unique_ptr<Window> m_Window;
    bool m_Running = true;
};

// 该函数由派生的应用定义
Application* CreateApplication();

}
