#include "path_tracer_layer.h"

#include "glad/gl.h"
#include "cuda_gl_interop.h"
#include "imgui.h"

#include "asteroid/base/application.h"
#include "asteroid/util/helper_cuda.h"

using namespace Asteroid;

extern "C" void launch_cudaProcess(dim3 grid, dim3 block, int sbytes,
    uchar4* g_odata, int imgw);

ExampleLayer::ExampleLayer()
    : Layer("Example")
{
    InitImage();

    InitVao();

    InitShader();

    InitFbo();

    InitCuda();

    m_Shader->Bind();
    m_Shader->UploadUniformInt("u_Texture", 0);
}

ExampleLayer::~ExampleLayer()
{
}

void ExampleLayer::OnUpdate()
{
    Render();

    Preview();
}

void ExampleLayer::OnImGuiRender()
{
    Application& app = Application::Get();
    auto width = (float)app.GetWindow().GetWidth();
    auto height = (float)app.GetWindow().GetHeight();

    ImGui::Begin("OpenGL Texture Text");
    ImGui::Text("pointer = %p", m_Texture->GetRendererID());
    ImGui::Text("size = %f x %f", width, height);
    ImGui::Image((void*)(intptr_t)m_Texture->GetRendererID(), ImVec2(width, height));
    ImGui::End();
}

void ExampleLayer::OnEvent(Event &event)
{
}

void ExampleLayer::InitImage()
{
    Application& app = Application::Get();
    auto width = app.GetWindow().GetWidth();
    auto height = app.GetWindow().GetHeight();

    TextureSpecification texSpec{};
    texSpec.Width = width;
    texSpec.Height = height;
    texSpec.Format = InternalFormat::RGBA8;
    texSpec.pixel_format = PixelFormat::RGBA;
    texSpec.GenerateMips = false;

    m_Image = std::make_shared<Image>(texSpec);

    m_Texture = std::make_shared<Texture2D>("D:/Learning/asteroid/asset/texture/bennu_dec10.png");
}

void ExampleLayer::InitShader()
{
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

    m_Shader = std::make_shared<Shader>(textureShaderVertexSrc, textureShaderFragmentSrc);

}

void ExampleLayer::InitVao()
{
    m_Vao = std::make_shared<VertexArray>();

    float squareVertices[5 * 4] = {
        -1.f, -1.f, 0.f, 0.f, 0.f,
        1.f, -1.f, 0.f, 1.f, 0.f,
        1.f, 1.f, 0.f, 1.f, 1.f,
        -1.f, 1.f, 0.f, 0.f, 1.f
    };

    auto squareVB = std::make_shared<VertexBuffer>(squareVertices, sizeof(squareVertices));
    squareVB->SetLayout({ { ShaderDataType::Float3, "a_Position" },
                         { ShaderDataType::Float2, "a_TexCoord" } });
    m_Vao->AddVertexBuffer(squareVB);

    unsigned int squareIndices[6] = { 0, 1, 3, 3, 1, 2 };
    auto squareIB = std::make_shared<IndexBuffer>(squareIndices, sizeof(squareIndices) / sizeof(unsigned int));
    m_Vao->SetIndexBuffer(squareIB);
}

void ExampleLayer::InitFbo()
{
    Application& app = Application::Get();
    auto width = app.GetWindow().GetWidth();
    auto height = app.GetWindow().GetHeight();

    FramebufferSpecification spec{};
    spec.Width = width;
    spec.Height = height;
    m_Fbo = std::make_shared<Framebuffer>(spec);
}

void ExampleLayer::InitCuda()
{
    Application& app = Application::Get();
    auto width = app.GetWindow().GetWidth();
    auto height = app.GetWindow().GetHeight();
    int num_texels = width * height;
    int num_values = num_texels * 4;
    int size_tex_data = sizeof(unsigned char) * num_values;
    checkCudaErrors(cudaMalloc((void**)&m_ImageData, size_tex_data));
}

void ExampleLayer::Render()
{
    Application& app = Application::Get();
    auto width = app.GetWindow().GetWidth();
    auto height = app.GetWindow().GetHeight();

    // Execute the kernel
    dim3 block(16, 16, 1);
    dim3 grid(width / block.x, height / block.y, 1);
    launch_cudaProcess(grid, block, 0, m_ImageData, width);
}

void ExampleLayer::Preview()
{
    Application& app = Application::Get();
    auto width = app.GetWindow().GetWidth();
    auto height = app.GetWindow().GetHeight();

    // set up vertex data parameter
    int num_texels = width * height;
    int num_values = num_texels * 4;
    int size_tex_data = sizeof(unsigned char) * num_values;

    m_Image->SetData(m_ImageData, size_tex_data);
    m_Texture->Bind();
    m_Vao->Bind();
    m_Shader->Bind();
    m_Shader->UploadUniformInt("u_Texture", 0);
    m_Fbo->Bind();
    glDrawElements(GL_TRIANGLES, m_Vao->GetIndexBuffer()->GetCount(), GL_UNSIGNED_INT, 0);
    m_Fbo->Unbind();
    glBindTexture(GL_TEXTURE_2D, m_Fbo->GetColorAttachmentRendererID());
    glDrawElements(GL_TRIANGLES, m_Vao->GetIndexBuffer()->GetCount(), GL_UNSIGNED_INT, 0);
}

