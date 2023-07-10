#include "asteroid/input/input.h"

#include "GLFW/glfw3.h"
#include "asteroid/base/application.h"

using namespace Asteroid;

bool Input::IsKeyDown(KeyCode keycode)
{
    auto window = static_cast<GLFWwindow*>(Application::Get().GetWindow().GetNativeWindow());
    auto state = glfwGetKey(window, (int)keycode);
    return state == GLFW_PRESS || state == GLFW_REPEAT;
}

bool Input::IsMouseButtonDown(MouseButton button)
{
    auto window = static_cast<GLFWwindow*>(Application::Get().GetWindow().GetNativeWindow());
    auto state = glfwGetMouseButton(window, (int)button);
    return state == GLFW_PRESS;
}

glm::vec2 Input::GetMousePosition()
{
    auto window = static_cast<GLFWwindow*>(Application::Get().GetWindow().GetNativeWindow());
    double x, y;
    glfwGetCursorPos(window, &x, &y);

    return { (float)x, (float)y };
}

void Input::SetCursorMode(CursorMode mode)
{
    auto window = static_cast<GLFWwindow*>(Application::Get().GetWindow().GetNativeWindow());
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL + (int)mode);
}
