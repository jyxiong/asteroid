#pragma once

#include <utility>

#include "asteroid/core/core.h"
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

// Interface representing a desktop system based Window
class Window
{
public:
    using EventCallbackFn = std::function<void(Event &)>;

    virtual ~Window() = default;

    virtual void OnUpdate() = 0;

    virtual unsigned int GetWidth() const = 0;

    virtual unsigned int GetHeight() const = 0;

    // Window attributes
    virtual void SetEventCallback(const EventCallbackFn &callback) = 0;

    virtual void SetVSync(bool enabled) = 0;

    virtual bool IsVSync() const = 0;

    virtual void *GetNativeWindow() const = 0;

    static Window *Create(const WindowProps &props = WindowProps());
};

}