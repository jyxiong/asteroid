#pragma once

#include <filesystem>

class Shader
{
private:
    unsigned int m_id{};

public:
    Shader(const std::filesystem::path &vertexPath, const std::filesystem::path &fragmentPath);
    ~Shader();

    void bind();
};