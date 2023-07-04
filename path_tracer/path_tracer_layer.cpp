#include "path_tracer_layer.h"

#include "glad/gl.h"
#include "cuda_gl_interop.h"
#include "asteroid/base/application.h"

using namespace Asteroid;

extern "C" void launch_cudaProcess(dim3 grid, dim3 block, int sbytes,
                                   unsigned char *g_odata, int imgw);

ExampleLayer::ExampleLayer()
    : Layer("Example")
{
    InitVao();

    InitTexture();

    InitShader();

    m_TextureShader->Bind();
    m_TextureShader->UploadUniformInt("u_Texture", 0);
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
        -1.f, -1.f, 0.f, 0.f, 0.f,
        1.f, -1.f, 0.f, 1.f, 0.f,
        1.f, 1.f, 0.f, 1.f, 1.f,
        -1.f, 1.f, 0.f, 0.f, 1.f
    };

    auto squareVB = std::make_shared<VertexBuffer>(squareVertices, sizeof(squareVertices));
    squareVB->SetLayout({{ ShaderDataType::Float3, "a_Position" },
                         { ShaderDataType::Float2, "a_TexCoord" }});
    m_SquareVA->AddVertexBuffer(squareVB);

    unsigned int squareIndices[6] = { 0, 1, 2, 2, 3, 0 };
    auto squareIB = std::make_shared<IndexBuffer>(squareIndices, sizeof(squareIndices) / sizeof(unsigned int));
    m_SquareVA->SetIndexBuffer(squareIB);
}

void ExampleLayer::InitTexture()
{
    //m_Texture = std::make_shared<Texture2D>(R"(D:\Learning\asteroid\asset\texture\bennu_dec10.png)");

    Application& app = Application::Get();
    auto width = app.GetWindow().GetWidth();
    auto height = app.GetWindow().GetHeight();

    TextureSpecification texSpec{};
    texSpec.Width = width;
    texSpec.Height = height;
    texSpec.Format = ImageFormat::RGBA8UI;
    texSpec.GenerateMips = false;
    m_Texture = std::make_shared<Texture2D>(texSpec);


    // register this texture with CUDA
    cudaGraphicsGLRegisterImage(&m_CudaResource, m_Texture->GetRendererID(), GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard);
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
    cudaMalloc((void**)&m_Data, size_tex_data);
}

void ExampleLayer::UpdateCuda()
{
    Application& app = Application::Get();
    auto width = app.GetWindow().GetWidth();
    auto height = app.GetWindow().GetHeight();

    // run the Cuda kernel
    // calculate grid size
    dim3 block(16, 16, 1);
    // dim3 block(16, 16, 1);
    dim3 grid(width / block.x, height / block.y, 1);
    // execute CUDA kernel
    launch_cudaProcess(grid, block, 0, m_Data, width);

    // map the texture and blit the result thanks to CUDA API
    // We want to copy out_data data to the texture
    // map buffer objects to get CUDA device pointers
    cudaArray *texture_ptr;
    cudaGraphicsMapResources(1, &m_CudaResource, 0);
    cudaGraphicsSubResourceGetMappedArray(
        &texture_ptr, m_CudaResource, 0, 0);

    int num_texels = width * height;
    int num_values = num_texels * 4;
    int size_tex_data = sizeof(unsigned char) * num_values;
    cudaMemcpyToArray(texture_ptr, 0, 0, m_Data,
                                      size_tex_data, cudaMemcpyDeviceToDevice);

    cudaGraphicsUnmapResources(1, &m_CudaResource, 0);

}

void ExampleLayer::UpdateOpengl()
{
    glClearColor(0.1f, 0.1f, 0.1f, 1);
    glClear(GL_COLOR_BUFFER_BIT);

    m_Texture->Bind();
    m_TextureShader->Bind();
    m_SquareVA->Bind();
    glDrawElements(GL_TRIANGLES, m_SquareVA->GetIndexBuffer()->GetCount(), GL_UNSIGNED_INT, nullptr);
}

