#pragma once

#include "asteroid/base/layer.h"
#include "asteroid/event/application_event.h"
#include "asteroid/event/key_event.h"
#include "asteroid/event/mouse_event.h"

namespace Asteroid
{

class ImGuiLayer : public Layer
{
public:
    ImGuiLayer();

    ~ImGuiLayer() override;

    void OnAttach() override;

    void OnDetach() override;

    void OnImGuiRender() override;

    void Begin();

    void End();

private:
    float m_Time = 0.0f;
};

}