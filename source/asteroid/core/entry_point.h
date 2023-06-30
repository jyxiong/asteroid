#pragma once

#include "asteroid/core/application.h"
#include "asteroid/core/log.h"

extern Asteroid::Application *Asteroid::CreateApplication();

int main(int argc, char **argv)
{
    Asteroid::Log::Init();

    auto app = Asteroid::CreateApplication();
    app->Run();
    delete app;
}