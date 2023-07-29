#include "asteroid/base/application.h"
#include "asteroid/base/entry_point.h"
#include "optix_layer.h"

using namespace Asteroid;

class Sandbox : public Application
{
public:
    Sandbox()
    {
        PushLayer(new OptixLayer());
    }

    ~Sandbox() override = default;
};

Asteroid::Application *Asteroid::CreateApplication()
{
    return new Sandbox();
}