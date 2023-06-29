#pragma once

#include <cstdio>
#include <iostream>
#include "glad/gl.h"
#include "GLFW/glfw3.h"

void errorCallback(int error, const char *description) {
    fprintf(stderr, "error %d: %s\n", error, description);
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GL_TRUE);
    }
}

class Application
{
private:
    GLFWwindow* m_window;

public:
    Application()
    {
    }

    void init()
    {
        glfwSetErrorCallback(errorCallback);

        glfwInit();
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

        m_window = glfwCreateWindow(960, 640, "Simple example", nullptr, nullptr);
        if (!m_window)
        {
            glfwTerminate();
            exit(EXIT_FAILURE);
        }
        
        glfwMakeContextCurrent(m_window);
        gladLoadGL(glfwGetProcAddress);
        glfwSwapInterval(1);

        glfwSetKeyCallback(m_window, keyCallback);
    }

    void run()
    {
        while (!glfwWindowShouldClose(m_window))
        {
            glfwSwapBuffers(m_window);
            glfwPollEvents();
        }

        glfwTerminate();
    }

};