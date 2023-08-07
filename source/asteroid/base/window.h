#pragma once

#include <string>
#include <functional>
#include "glad/gl.h"
#include "GLFW/glfw3.h"
#include "asteroid/event/event.h"

namespace Asteroid
{

struct WindowProps
{
    std::string Title;
    unsigned int Width;
    unsigned int Height;

    explicit WindowProps(std::string title = "Asteroid Engine",
                         unsigned int width = 1280,
                         unsigned int height = 720)
        : Title(std::move(title)), Width(width), Height(height)
    {
    }
};

class Window
{
public:
    using EventCallbackFn = std::function<void(Event &)>;

    explicit Window(const WindowProps &props);

    ~Window();

    void OnUpdate();

    inline unsigned int GetWidth() const { return m_Data.Width; }

    inline unsigned int GetHeight() const { return m_Data.Height; }

    // Window attributes
    inline void SetEventCallback(const EventCallbackFn &callback) { m_Data.EventCallback = callback; }

    inline void* GetNativeWindow() const { return m_Window; }

    static Window *Create(const WindowProps &props = WindowProps());

private:
    GLFWwindow *m_Window{};

    struct WindowData
    {
        std::string Title;
        unsigned int Width, Height;

        EventCallbackFn EventCallback;
    };

    WindowData m_Data;
};

}