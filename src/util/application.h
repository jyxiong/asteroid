#pragma once

#include <filesystem>
#include <memory>
#include "glad/gl.h"
#include "GLFW/glfw3.h"

class Application
{
public:
    static const std::filesystem::path &getDataPath() { return s_dataPath; }
    static void setDataPath(const std::filesystem::path &path) { s_dataPath = path; };

    static void awake();
    static void run();

    static void update();
    static void updateScreenSize();

    static void render();

private:
    static void initOpenGl();

public:
    static std::filesystem::path s_dataPath;

    static GLFWwindow *s_window;

}; // class Application
