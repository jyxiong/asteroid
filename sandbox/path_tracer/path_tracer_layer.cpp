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
}

void ExampleLayer::OnImGuiRender()
{
    // Note: Switch this to true to enable dockspace
    static bool dockspaceOpen = true;
    static bool opt_fullscreen_persistant = true;
    bool opt_fullscreen = opt_fullscreen_persistant;
    static ImGuiDockNodeFlags dockspace_flags = ImGuiDockNodeFlags_None;

    // We are using the ImGuiWindowFlags_NoDocking flag to make the parent window not dockable into,
    // because it would be confusing to have two docking targets within each others.
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking;
    if (opt_fullscreen)
    {
        ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(viewport->Pos);
        ImGui::SetNextWindowSize(viewport->Size);
        ImGui::SetNextWindowViewport(viewport->ID);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
        window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;
    }

    // When using ImGuiDockNodeFlags_PassthruCentralNode, DockSpace() will render our background and handle the pass-thru hole, so we ask Begin() to not render a background.
    if (dockspace_flags & ImGuiDockNodeFlags_PassthruCentralNode)
        window_flags |= ImGuiWindowFlags_NoBackground;

    // Important: note that we proceed even if Begin() returns false (aka window is collapsed).
    // This is because we want to keep our DockSpace() active. If a DockSpace() is inactive, 
    // all active windows docked into it will lose their parent and become undocked.
    // We cannot preserve the docking relationship between an active window and an inactive docking, otherwise 
    // any change of dockspace/settings would lead to windows being stuck in limbo and never being visible.
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::Begin("DockSpace Demo", &dockspaceOpen, window_flags);
    ImGui::PopStyleVar();

    if (opt_fullscreen)
        ImGui::PopStyleVar(2);

    // DockSpace
    ImGuiIO& io = ImGui::GetIO();
    ImGuiStyle& style = ImGui::GetStyle();
    float minWinSizeX = style.WindowMinSize.x;
    style.WindowMinSize.x = 370.0f;
    if (io.ConfigFlags & ImGuiConfigFlags_DockingEnable)
    {
        ImGuiID dockspace_id = ImGui::GetID("MyDockSpace");
        ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), dockspace_flags);
    }

    style.WindowMinSize.x = minWinSizeX;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2{ 0, 0 });
    ImGui::Begin("Viewport");

    Application& app = Application::Get();
    auto width = (float)app.GetWindow().GetWidth();
    auto height = (float)app.GetWindow().GetHeight();

    ImGui::Image(reinterpret_cast<void*>(m_Image->m_Texture->GetRendererID()), ImVec2{ width, height }, ImVec2{ 0, 1 }, ImVec2{ 1, 0 });

    ImGui::End();
    ImGui::PopStyleVar();

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
        -1.f, -1.f, 0.f, 1.f, 1.f,
        1.f, -1.f, 0.f, 0.f, 1.f,
        1.f, 1.f, 0.f, 0.f, 0.f,
        -1.f, 1.f, 0.f, 1.f, 0.f
    };

    auto squareVB = std::make_shared<VertexBuffer>(squareVertices, sizeof(squareVertices));
    squareVB->SetLayout({ { ShaderDataType::Float3, "a_Position" },
                         { ShaderDataType::Float2, "a_TexCoord" } });
    m_Vao->AddVertexBuffer(squareVB);

    unsigned int squareIndices[6] = { 0, 1, 3, 3, 1, 2 };
    auto squareIB = std::make_shared<IndexBuffer>(squareIndices, sizeof(squareIndices) / sizeof(unsigned int));
    m_Vao->SetIndexBuffer(squareIB);
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
    m_Image->m_Texture->Bind();
    m_Vao->Bind();
    m_Shader->Bind();
    glDrawElements(GL_TRIANGLES, m_Vao->GetIndexBuffer()->GetCount(), GL_UNSIGNED_INT, 0);
}

