#pragma once

#include <unordered_map>

#include "glm/glm.hpp"

#include "control/keycode.h"

class Input
{
public:
    static void recordKey(KeyCode keyCode, KeyAction keyAction);
    static bool isKey(KeyCode keyCode);
    static bool isKeyDown(KeyCode keyCode);
    static bool isKeyUp(KeyCode keyCode);

    static void recordMouseButton(MouseButton mouseButton, MouseButtonAction mouseButtonAction);
    static bool isMouseButton(MouseButton mouseButton);
    static bool isMouseButtonDown(MouseButton mouseButton);
    static bool isMouseButtonUp(MouseButton mouseButton);

    static void recordMousePosition(float x, float y);
    static glm::vec2 getMousePosition() { return s_mousePosition; };

    static void recordMouseScroll(float mouseScroll);
    static float getMouseScroll() { return s_mouseScroll; }

    static void update();

private:
    static std::unordered_map<KeyCode, KeyAction> s_keyEvents;
    static std::unordered_map<MouseButton, MouseButtonAction> s_mouseButtonEvents;

    static glm::vec2 s_mousePosition;
    static float s_mouseScroll;

}; // class Input
