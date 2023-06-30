#include "asteroid/core/application.h"
#include "asteroid/core/entry_point.h"

using namespace Asteroid;

class Sandbox : public Application
{
public:
    Sandbox()
    {

    }

    ~Sandbox()
    {

    }
};

Asteroid::Application *Asteroid::CreateApplication()
{
    return new Sandbox();
}