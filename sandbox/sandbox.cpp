#include "asteroid/core/application.h"
#include "asteroid/core/entry_point.h"
#include "asteroid/core/layer.h"
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
        AST_INFO("ExampleLayer::Update");
    }

    void OnEvent(Event& event) override
    {
        AST_TRACE("{0}", event.ToString());
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