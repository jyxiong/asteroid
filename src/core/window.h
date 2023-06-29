#pragma once

#include <string>
#include "Glfw/glfw3.h"

class Window
{
public:
    Window();

    Window(int width, int height, std::string title);

    ~Window();

    bool shouldClose() { return glfwWindowShouldClose(m_window); }

private:
    void initWindow();

    int m_width{};
    int m_height{};

    std::string m_title{};

    GLFWwindow *m_window{};

}; // class Window