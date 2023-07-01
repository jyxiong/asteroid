#pragma once

#include "asteroid/core/layer.h"
#include "asteroid/event/application_event.h"
#include "asteroid/event/key_event.h"
#include "asteroid/event/mouse_event.h"

namespace Asteroid
{

class ImGuiLayer : public Layer
{
public:
    ImGuiLayer();

    ~ImGuiLayer();

    void OnAttach();

    void OnDetach();

    void OnUpdate();

    void OnEvent(Event &event);

private:
    bool OnMouseButtonPressedEvent(MouseButtonPressedEvent& e);
    bool OnMouseButtonReleasedEvent(MouseButtonReleasedEvent& e);
    bool OnMouseMovedEvent(MouseMovedEvent& e);
    bool OnMouseScrolledEvent(MouseScrolledEvent& e);
    bool OnKeyPressedEvent(KeyPressedEvent& e);
    bool OnKeyReleasedEvent(KeyReleasedEvent& e);
    bool OnKeyTypedEvent(KeyTypedEvent& e);
    bool OnWindowResizeEvent(WindowResizeEvent& e);

private:
    float m_Time = 0.0f;
};

}