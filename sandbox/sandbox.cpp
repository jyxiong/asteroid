#include "imgui.h"

#include "asteroid/base/application.h"
#include "asteroid/base/entry_point.h"
#include "asteroid/base/layer.h"
#include "asteroid/imgui/imgui_layer.h"
#include "asteroid/opengl/shader.h"
#include "asteroid/opengl/vertex_buffer.h"
#include "asteroid/opengl/vertex_array.h"
#include "asteroid/opengl/texture2d.h"

using namespace Asteroid;

class ExampleLayer : public Layer
{
public:
    ExampleLayer()
        : Layer("Example")
    {
        m_SquareVA = std::make_shared<VertexArray>();

        float squareVertices[5 * 4] = {
            -0.5f, -0.5f, 0.0f, 0.0f, 0.0f,
            0.5f, -0.5f, 0.0f, 1.0f, 0.0f,
            0.5f, 0.5f, 0.0f, 1.0f, 1.0f,
            -0.5f, 0.5f, 0.0f, 0.0f, 1.0f
        };

        auto squareVB = std::make_shared<VertexBuffer>(squareVertices, sizeof(squareVertices));
        squareVB->SetLayout({{ ShaderDataType::Float3, "a_Position" },
                             { ShaderDataType::Float2, "a_TexCoord" }});
        m_SquareVA->AddVertexBuffer(squareVB);

        unsigned int squareIndices[6] = { 0, 1, 2, 2, 3, 0 };
        auto squareIB = std::make_shared<IndexBuffer>(squareIndices, sizeof(squareIndices) / sizeof(unsigned int));
        m_SquareVA->SetIndexBuffer(squareIB);

        std::string textureShaderVertexSrc = R"(
			#version 330 core

            layout(location = 0) in vec3 a_Position;
			layout(location = 1) in vec2 a_TexCoord;
			out vec2 v_TexCoord;
			void main()
			{
				v_TexCoord = a_TexCoord;
				gl_Position = vec4(a_Position, 1.0);
			}
		)";

        std::string textureShaderFragmentSrc = R"(
			#version 330 core

            layout(location = 0) out vec4 color;
			in vec2 v_TexCoord;

			uniform sampler2D u_Texture;
			void main()
			{
				color = texture(u_Texture, v_TexCoord);
			}
		)";

        m_TextureShader = std::make_shared<Shader>(textureShaderVertexSrc, textureShaderFragmentSrc);

        m_Texture = std::make_shared<Texture2D>(R"(D:\Learning\asteroid\asset\texture\bennu_dec10.png)");

        m_TextureShader->Bind();
        m_TextureShader->UploadUniformInt("u_Texture", 0);
    }

    void OnUpdate() override
    {
        glClearColor(0.1f, 0.1f, 0.1f, 1);
        glClear(GL_COLOR_BUFFER_BIT);

        m_Texture->Bind();
        m_TextureShader->Bind();
        m_SquareVA->Bind();
        glDrawElements(GL_TRIANGLES, m_SquareVA->GetIndexBuffer()->GetCount(), GL_UNSIGNED_INT, nullptr);
    }

    void OnImGuiRender() override
    {
    }

    void OnEvent(Event &event) override
    {
    }

private:
    std::shared_ptr<Shader> m_TextureShader;
    std::shared_ptr<VertexArray> m_SquareVA;

    std::shared_ptr<Texture2D> m_Texture;
};

class Sandbox : public Application
{
public:
    Sandbox()
    {
        PushLayer(new ExampleLayer());
    }

    ~Sandbox()
    {

    }
};

Asteroid::Application *Asteroid::CreateApplication()
{
    return new Sandbox();
}