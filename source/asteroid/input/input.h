#pragma once

#include <cuda_runtime.h>
#include "asteroid/input/key_code.h"

namespace Asteroid
{

class Input
{
public:
    static bool IsKeyDown(KeyCode keycode);
    static bool IsMouseButtonDown(MouseButton button);

    static float2 GetMousePosition();

    static void SetCursorMode(CursorMode mode);
};

}
