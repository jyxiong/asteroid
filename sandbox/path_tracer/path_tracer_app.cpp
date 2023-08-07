#include "asteroid/app/application.h"
#include "asteroid/app/entry_point.h"
#include "path_tracer_layer.h"

using namespace Asteroid;

class Sandbox : public Application
{
public:
    Sandbox()
    {
        PushLayer(new ExampleLayer());
    }

    ~Sandbox()
    {

    }
};

Asteroid::Application *Asteroid::CreateApplication()
{
    return new Sandbox();
}