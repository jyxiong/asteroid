#include "Application.h"

#include "control/input.h"
#include "util/log.h"

#include "Screen.h"

static void errorCallback(int error, const char *description)
{
    fprintf(stderr, "Error: %s\n", description);
}

static void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
    Input::recordKey(static_cast<KeyCode>(key), static_cast<KeyAction>(action));
}

static void mouseButtonCallback(GLFWwindow *window, int button, int action, int mods)
{
    Input::recordMouseButton(static_cast<MouseButton>(button), static_cast<MouseButtonAction>(action));
}

static void mouseMoveCallback(GLFWwindow *window, double x, double y)
{
    Input::recordMousePosition(x, y);
}

static void mouseScrollCallback(GLFWwindow *window, double x, double y)
{
    Input::recordMouseScroll(y);
}

std::filesystem::path Application::s_dataPath;
GLFWwindow *Application::s_window;

void Application::awake()
{
    Log::init();
    initOpenGl();
}

void Application::run()
{
    while (!glfwWindowShouldClose(s_window))
    {
        update();
        render();

        glfwSwapBuffers(s_window);

        glfwPollEvents();
    }

    glfwTerminate();
}

void Application::update()
{
    updateScreenSize();

    Input::update();
}

void Application::updateScreenSize()
{
    int width, height;
    glfwGetFramebufferSize(s_window, &width, &height);
    glViewport(0, 0, width, height);

    Screen::setSize(width, height);
}

void Application::render()
{
}

void Application::initOpenGl()
{
    glfwSetErrorCallback(errorCallback);

    if (!glfwInit())
        exit(EXIT_FAILURE);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    s_window = glfwCreateWindow(960, 640, "Simple example", nullptr, nullptr);
    if (!s_window)
    {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    glfwMakeContextCurrent(s_window);
    gladLoadGL(glfwGetProcAddress);
    glfwSwapInterval(1);

    glfwSetKeyCallback(s_window, keyCallback);
    glfwSetMouseButtonCallback(s_window, mouseButtonCallback);
    glfwSetScrollCallback(s_window, mouseScrollCallback);
    glfwSetCursorPosCallback(s_window, mouseMoveCallback);
}
