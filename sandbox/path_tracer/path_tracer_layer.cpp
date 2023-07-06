#include "path_tracer_layer.h"

#include "glad/gl.h"
#include "cuda_gl_interop.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "asteroid/base/application.h"
#include "asteroid/util/helper_cuda.h"

using namespace Asteroid;

extern "C" void launch_cudaProcess(dim3 grid, dim3 block, int sbytes,
    uchar4* g_odata, int imgw);

ExampleLayer::ExampleLayer()
    : Layer("Example")
{
    InitVao();

    InitTexture();

    InitShader();

    InitCuda();

    InitPbo();

    m_TextureShader->Bind();
    m_TextureShader->UploadUniformInt("u_Texture", 0);
}

ExampleLayer::~ExampleLayer()
{
}

void ExampleLayer::OnUpdate()
{
    UpdateCuda();

    UpdateOpengl();
}

void ExampleLayer::OnImGuiRender()
{
}

void ExampleLayer::OnEvent(Event &event)
{
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

    m_TextureShader = std::make_shared<Shader>(textureShaderVertexSrc, textureShaderFragmentSrc);

}

void ExampleLayer::InitVao()
{
    m_SquareVA = std::make_shared<VertexArray>();

    float squareVertices[5 * 4] = {
        -1.f, -1.f, 0.f, 1.f, 1.f,
        1.f, -1.f, 0.f, 0.f, 1.f,
        1.f, 1.f, 0.f, 0.f, 0.f,
        -1.f, 1.f, 0.f, 1.f, 0.f
    };

    auto squareVB = std::make_shared<VertexBuffer>(squareVertices, sizeof(squareVertices));
    squareVB->SetLayout({ { ShaderDataType::Float3, "a_Position" },
                         { ShaderDataType::Float2, "a_TexCoord" } });
    m_SquareVA->AddVertexBuffer(squareVB);

    unsigned int squareIndices[6] = { 0, 1, 3, 3, 1, 2 };
    auto squareIB = std::make_shared<IndexBuffer>(squareIndices, sizeof(squareIndices) / sizeof(unsigned int));
    m_SquareVA->SetIndexBuffer(squareIB);
}

void ExampleLayer::InitPbo()
{
    Application& app = Application::Get();
    auto width = app.GetWindow().GetWidth();
    auto height = app.GetWindow().GetHeight();

    int num_texels = width * height;
    int num_values = num_texels * 4;
    size_t size_tex_data = sizeof(GLubyte) * num_values;

    m_Pbo = std::make_shared<PixelBuffer>(size_tex_data);

    // 注册到cuda
    cudaGraphicsGLRegisterBuffer(&m_resource, m_Pbo->GetRendererID(), cudaGraphicsMapFlagsNone);
}

void ExampleLayer::InitTexture()
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
    m_Texture = std::make_shared<Texture2D>(texSpec);
}

void ExampleLayer::InitCuda()
{
    Application& app = Application::Get();
    auto width = app.GetWindow().GetWidth();
    auto height = app.GetWindow().GetHeight();

    // set up vertex data parameter
    int num_texels = width * height;
    int num_values = num_texels * 4;
    int size_tex_data = sizeof(unsigned char) * num_values;
    //checkCudaErrors(cudaMalloc((void**)&m_Data, size_tex_data));
}

void ExampleLayer::UpdateCuda()
{
    Application& app = Application::Get();
    auto width = app.GetWindow().GetWidth();
    auto height = app.GetWindow().GetHeight();

    size_t num_bytes;

    uchar4* dptr = NULL;
    cudaGraphicsMapResources(1, &m_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes, m_resource);

    // Execute the kernel
    dim3 block(16, 16, 1);
    dim3 grid(width / block.x, height / block.y, 1);
    launch_cudaProcess(grid, block, 0, dptr, width);

    // Unmap buffer object
    cudaGraphicsUnmapResources(1, &m_resource, 0);
}

void ExampleLayer::UpdateOpengl()
{
    // https://www.cnblogs.com/crsky/p/7870835.html
    m_Pbo->BindUnpack();
    m_Texture->Bind();
    m_Texture->SetData(nullptr, 0);
    m_SquareVA->Bind();
    m_TextureShader->Bind();
    glDrawElements(GL_TRIANGLES, m_SquareVA->GetIndexBuffer()->GetCount(), GL_UNSIGNED_INT, 0);
}

