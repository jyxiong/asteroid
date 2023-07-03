#pragma once

#include <string>
#include <filesystem>

namespace Asteroid
{

    class Shader
    {
    public:
        Shader(const std::filesystem::path &vertexPath, const std::filesystem::path &fragmentPath);

        Shader(const std::string &vertexSrc, const std::string &fragmentSrc);

        ~Shader();

        void Bind() const;

        void Unbind() const;

        void UploadUniformInt(const std::string& name, int value);

    private:
        uint32_t m_RendererID;
    };

}