#pragma once

#include "asteroid/app/application.h"
#include "asteroid/util/log.h"

extern Asteroid::Application *Asteroid::CreateApplication();

int main(int argc, char **argv)
{
    Asteroid::Log::Init();

    auto app = Asteroid::CreateApplication();
    app->Run();
    delete app;
}