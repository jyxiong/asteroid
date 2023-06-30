#include "control/input.h"

std::unordered_map<KeyCode, KeyAction> Input::s_keyEvents;
std::unordered_map<MouseButton, MouseButtonAction> Input::s_mouseButtonEvents;
glm::vec2 Input::s_mousePosition{ 0 };
float Input::s_mouseScroll{ 0 };

void Input::recordKey(KeyCode keyCode, KeyAction keyAction)
{
    s_keyEvents[keyCode] = keyAction;
}

void Input::recordMouseButton(MouseButton mouseButton, MouseButtonAction mouseButtonAction)
{
    s_mouseButtonEvents[mouseButton] = mouseButtonAction;
}

bool Input::isKey(KeyCode keyCode)
{
    return s_keyEvents.count(keyCode) > 0;
}

bool Input::isKeyDown(KeyCode keyCode)
{
    return s_keyEvents.count(keyCode) != 0 && s_keyEvents[keyCode] != KeyAction::Up;
}

bool Input::isKeyUp(KeyCode keyCode)
{
    return s_keyEvents.count(keyCode) != 0 && s_keyEvents[keyCode] == KeyAction::Up;
}

bool Input::isMouseButton(MouseButton mouseButton)
{
    return s_mouseButtonEvents.count(mouseButton) > 0;
}

bool Input::isMouseButtonDown(MouseButton mouseButton)
{
    return s_mouseButtonEvents.count(mouseButton) != 0 && s_mouseButtonEvents[mouseButton] != MouseButtonAction::Up;
}

bool Input::isMouseButtonUp(MouseButton mouseButton)
{
    return s_mouseButtonEvents.count(mouseButton) != 0 && s_mouseButtonEvents[mouseButton] == MouseButtonAction::Up;
}

void Input::recordMousePosition(float x, float y)
{
    s_mousePosition.x = x;
    s_mousePosition.y = y;
}

void Input::recordMouseScroll(float mouseScroll)
{
    s_mouseScroll += mouseScroll;
}

void Input::update()
{
    for (auto iter = s_keyEvents.begin(); iter != s_keyEvents.end();)
    {
        if (iter->second == KeyAction::Up)
        {
            iter = s_keyEvents.erase(iter);
        } else
        {
            ++iter;
        }
    }

    for (auto iter = s_mouseButtonEvents.begin(); iter != s_mouseButtonEvents.end();)
    {
        if (iter->second == MouseButtonAction::Up)
        {
            iter = s_mouseButtonEvents.erase(iter);
        } else
        {
            ++iter;
        }
    }

    s_mouseScroll = 0;
}
