#include "asteroid/app/window.h"
#include "asteroid/util/log.h"
#include "asteroid/util/macro.h"

using namespace Asteroid;

static bool s_GLFWInitialized = false;

static void GLFWErrorCallback(int error, const char *description)
{
    AST_CORE_ERROR("GLFW Error ({0}): {1}", error, description);
}

Window *Window::Create(const WindowProps &props)
{
    return new Window(props);
}

Window::Window(const WindowProps &props)
{
    m_Data.Title = props.Title;
    m_Data.Width = props.Width;
    m_Data.Height = props.Height;

    AST_CORE_INFO("Creating window {0} ({1}, {2})", props.Title, props.Width, props.Height);

    if (!s_GLFWInitialized)
    {
        // TODO: glfwTerminate on system shutdown
        auto success = glfwInit();
        AST_ASSERT(success, "Could not initialize GLFW!")

        s_GLFWInitialized = true;
    }

    m_Window = glfwCreateWindow((int) props.Width, (int) props.Height, m_Data.Title.c_str(), nullptr, nullptr);
    glfwMakeContextCurrent(m_Window);

    auto status = gladLoadGL(glfwGetProcAddress);
    AST_ASSERT(status, "Failed to initialize Glad!")

    AST_CORE_INFO("OpenGL Info:");
    AST_CORE_INFO("  Vendor: {}", fmt::ptr(glGetString(GL_VENDOR)));
    AST_CORE_INFO("  Renderer: {}", fmt::ptr(glGetString(GL_RENDERER)));
    AST_CORE_INFO("  Version: {}", fmt::ptr(glGetString(GL_VERSION)));

    glfwSetWindowUserPointer(m_Window, &m_Data);

    glfwSwapInterval(1);
}

Window::~Window()
{
    glfwDestroyWindow(m_Window);
}

void Window::OnUpdate()
{
    glfwPollEvents();
    glfwSwapBuffers(m_Window);
}
