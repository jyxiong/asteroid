#include "asteroid/core/application.h"
#include "asteroid/core/entry_point.h"
#include "asteroid/core/layer.h"
#include "asteroid/core/input.h"
#include "asteroid/core/key_code.h"
#include "asteroid/imgui/imgui_layer.h"

using namespace Asteroid;

class ExampleLayer : public Layer
{
public:
    ExampleLayer()
        : Layer("Example")
    {
    }

    void OnUpdate() override
    {
        if (Input::IsKeyPressed(AST_KEY_TAB))
            AST_TRACE("Tab key is pressed (poll)!");
    }

    void OnEvent(Event& event) override
    {
        if (event.GetEventType() == EventType::KeyPressed)
        {
            KeyPressedEvent& e = (KeyPressedEvent&)event;
            if (e.GetKeyCode() == AST_KEY_TAB)
                AST_TRACE("Tab key is pressed (event)!");
            AST_TRACE("{0}", (char)e.GetKeyCode());
        }
    }

};

class Sandbox : public Application
{
public:
    Sandbox()
    {
        PushLayer(new ExampleLayer());
        PushOverlay(new ImGuiLayer());
    }

    ~Sandbox()
    {

    }
};

Asteroid::Application *Asteroid::CreateApplication()
{
    return new Sandbox();
}