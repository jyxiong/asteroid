#pragma once

#include "asteroid/event/event.h"
#include "asteroid/event/application_event.h"
#include "asteroid/core/window.h"
#include "asteroid/core/layer.h"
#include "asteroid/core/layer_stack.h"
#include "asteroid/imgui/imgui_layer.h"
#include "asteroid/opengl/shader.h"
#include "asteroid/opengl/buffer.h"
#include "asteroid/opengl/vertex_array.h"

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
    ImGuiLayer* m_ImGuiLayer;
    bool m_Running = true;

    LayerStack m_LayerStack;

    std::shared_ptr<Shader> m_Shader;
    std::shared_ptr<VertexArray> m_VertexArray;

    std::shared_ptr<Shader> m_BlueShader;
    std::shared_ptr<VertexArray> m_SquareVA;

private:
    static Application* s_Instance;
};

// 该函数由派生的应用定义
Application* CreateApplication();

}
