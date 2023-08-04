#pragma once

#include "glm/glm.hpp"
#include "asteroid/input/key_code.h"

namespace Asteroid
{

class Input
{
public:
    static bool IsKeyDown(KeyCode keycode);
    static bool IsMouseButtonDown(MouseButton button);

    static glm::vec2 GetMousePosition();

    static void SetCursorMode(CursorMode mode);
};

}
