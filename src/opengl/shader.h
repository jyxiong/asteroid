#pragma once

#include <filesystem>
#include <fstream>
#include <string>
#include "glad/gl.h"

class Shader
{
private:
    unsigned int m_id{};

public:
    Shader(const std::filesystem::path &vertexPath, const std::filesystem::path &fragmentPath)
    {
        std::ifstream vsStream(vertexPath.string() + ".vs");
        std::string vs((std::istreambuf_iterator<char>(vsStream)), std::istreambuf_iterator<char>());

        std::ifstream fsStream(fragmentPath.string() + ".fs");
        std::string fs((std::istreambuf_iterator<char>(fsStream)), std::istreambuf_iterator<char>());

        const char *vsCode = vs.c_str();
        unsigned int vertex = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertex, 1, &vsCode, nullptr);
        glCompileShader(vertex);
        int compile_status = GL_FALSE;
        glGetShaderiv(vertex, GL_COMPILE_STATUS, &compile_status);
        if (compile_status == GL_FALSE)
        {
            char message[256];
            glGetShaderInfoLog(vertex, sizeof(message), nullptr, message);
        }

        const char *fsCode = fs.c_str();
        unsigned int fragment = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragment, 1, &fsCode, nullptr);
        glCompileShader(fragment);
        glGetShaderiv(fragment, GL_COMPILE_STATUS, &compile_status);
        if (compile_status == GL_FALSE)
        {
            char message[256];
            glGetShaderInfoLog(fragment, sizeof(message), nullptr, message);
        }

        m_id = glCreateProgram();
        glAttachShader(m_id, vertex);
        glAttachShader(m_id, fragment);
        glLinkProgram(m_id);
        int link_status = GL_FALSE;
        glGetProgramiv(m_id, GL_LINK_STATUS, &link_status);
        if (link_status == GL_FALSE)
        {
            char message[256];
            glGetProgramInfoLog(m_id, sizeof(message), nullptr, message);
        }

        glDeleteShader(vertex);
        glDeleteShader(fragment);
    }

    ~Shader()
    {
        glDeleteProgram(m_id);
    }

    void bind()
    {
        glUseProgram(m_id);
    }
};