#include "asteroid/app/application.h"
#include "asteroid/app/entry_point.h"
#include "optix_layer.h"

using namespace Asteroid;

class Sandbox : public Application
{
public:
    Sandbox()
    {
        PushLayer(std::make_shared<OptixLayer>());
    }

    ~Sandbox() override = default;
};

Asteroid::Application *Asteroid::CreateApplication()
{
    return new Sandbox();
}