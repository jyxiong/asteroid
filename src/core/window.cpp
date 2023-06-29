#include "window.h"

#include "log.h"

void errorCallback(int error, const char *description) {
    fprintf(stderr, "error %d: %s\n", error, description);
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GL_TRUE);
    }
}

Window::Window() = default;

Window::Window(int width, int height, std::string title)
    : m_width(width), m_height(height), m_title(std::move(title))
{
    initWindow();
}

Window::~Window()
{
    glfwDestroyWindow(m_window);
    glfwTerminate();
}

void Window::initWindow()
{
    glfwSetErrorCallback(errorCallback);

    glfwInit();
    glfwInitHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwInitHint(GLFW_RESIZABLE, GLFW_FALSE);

    m_window = glfwCreateWindow(m_width, m_height, m_title.c_str(), nullptr, nullptr);
    if (!m_window)
    {
        LOG_CORE_CRITICAL("failed to create glfw window!");
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    glfwSetKeyCallback(m_window, keyCallback);
}
