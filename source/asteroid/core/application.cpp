#include <memory>

#include "asteroid/core/application.h"
#include "glad/gl.h"
#include "asteroid/event/application_event.h"
#include "asteroid/core/log.h"
#include "asteroid/core/core.h"

using namespace Asteroid;

Application *Application::s_Instance = nullptr;

Application::Application()
{
    AST_CORE_ASSERT(!s_Instance, "Application already exists!")
    s_Instance = this;

    m_Window = std::unique_ptr<Window>(Window::Create());

    m_Window->SetEventCallback([this](Event &e) -> void {
        OnEvent(e);
    });

    m_ImGuiLayer = new ImGuiLayer();
    PushOverlay(m_ImGuiLayer);

    m_VertexArray = std::make_shared<VertexArray>();

    float vertices[3 * 7] = {
        -0.5f, -0.5f, 0.0f, 0.8f, 0.2f, 0.8f, 1.0f,
        0.5f, -0.5f, 0.0f, 0.2f, 0.3f, 0.8f, 1.0f,
        0.0f, 0.5f, 0.0f, 0.8f, 0.8f, 0.2f, 1.0f
    };

    auto vertexBuffer = std::make_shared<VertexBuffer>(vertices, sizeof(vertices));
    BufferLayout layout = {
        { ShaderDataType::Float3, "a_Position" },
        { ShaderDataType::Float4, "a_Color" }
    };
    vertexBuffer->SetLayout(layout);
    m_VertexArray->AddVertexBuffer(vertexBuffer);

    unsigned int indices[3] = { 0, 1, 2 };
    auto indexBuffer = std::make_shared<IndexBuffer>(indices, sizeof(indices) / sizeof(unsigned int));
    m_VertexArray->SetIndexBuffer(indexBuffer);

    m_SquareVA = std::make_shared<VertexArray>();

    float squareVertices[3 * 4] = {
        -0.75f, -0.75f, 0.0f,
        0.75f, -0.75f, 0.0f,
        0.75f, 0.75f, 0.0f,
        -0.75f, 0.75f, 0.0f
    };

    auto squareVB = std::make_shared<VertexBuffer>(squareVertices, sizeof(squareVertices));
    squareVB->SetLayout({
                            { ShaderDataType::Float3, "a_Position" }
                        });
    m_SquareVA->AddVertexBuffer(squareVB);

    unsigned int squareIndices[6] = { 0, 1, 2, 2, 3, 0 };
    auto squareIB = std::make_shared<IndexBuffer>(squareIndices, sizeof(squareIndices) / sizeof(unsigned int));
    m_SquareVA->SetIndexBuffer(squareIB);

    std::string vertexSrc = R"(
			#version 330 core

			layout(location = 0) in vec3 a_Position;
            layout(location = 1) in vec4 a_Color;

			out vec3 v_Position;
            out vec4 v_Color;

			void main()
			{
				v_Position = a_Position;
                v_Color = a_Color;

                gl_Position = vec4(a_Position, 1.0);
			}
		)";

    std::string fragmentSrc = R"(
			#version 330 core

			layout(location = 0) out vec4 color;
			in vec3 v_Position;
            in vec4 v_Color;

            void main()
			{
				color = vec4(v_Position * 0.5 + 0.5, 1.0);
                color = v_Color;
			}
		)";

    m_Shader = std::make_shared<Shader>(vertexSrc, fragmentSrc);

    std::string blueShaderVertexSrc = R"(
			#version 330 core

			layout(location = 0) in vec3 a_Position;
			out vec3 v_Position;
			void main()
			{
				v_Position = a_Position;
				gl_Position = vec4(a_Position, 1.0);
			}
		)";

    std::string blueShaderFragmentSrc = R"(
			#version 330 core

			layout(location = 0) out vec4 color;
			in vec3 v_Position;
			void main()
			{
				color = vec4(0.2, 0.3, 0.8, 1.0);
			}
		)";

    m_BlueShader = std::make_shared<Shader>(blueShaderVertexSrc, blueShaderFragmentSrc);
}

Application::~Application() = default;

void Application::PushLayer(Layer *layer)
{
    m_LayerStack.PushLayer(layer);
    layer->OnAttach();
}

void Application::PushOverlay(Layer *layer)
{
    m_LayerStack.PushOverlay(layer);
    layer->OnAttach();
}


void Application::OnEvent(Event &e)
{
    EventDispatcher dispatcher(e);
    dispatcher.Dispatch<WindowCloseEvent>([this](WindowCloseEvent &e) -> bool {
        return OnWindowClose(e);
    });

    for (auto it = m_LayerStack.end(); it != m_LayerStack.begin();)
    {
        (*--it)->OnEvent(e);
        if (e.Handled)
            break;
    }
}

void Application::Run()
{
    while (m_Running)
    {
        glClearColor(0.1f, 0.1f, 0.1f, 1);
        glClear(GL_COLOR_BUFFER_BIT);

        m_BlueShader->Bind();
        m_SquareVA->Bind();
        glDrawElements(GL_TRIANGLES, m_SquareVA->GetIndexBuffer()->GetCount(), GL_UNSIGNED_INT, nullptr);

        m_Shader->Bind();
        m_VertexArray->Bind();
        glDrawElements(GL_TRIANGLES, m_VertexArray->GetIndexBuffer()->GetCount(), GL_UNSIGNED_INT, nullptr);

        for (Layer *layer: m_LayerStack)
            layer->OnUpdate();

        m_ImGuiLayer->Begin();
        for (Layer *layer: m_LayerStack)
            layer->OnImGuiRender();
        m_ImGuiLayer->End();

        m_Window->OnUpdate();
    }
}

bool Application::OnWindowClose(WindowCloseEvent &e)
{
    m_Running = false;
    return true;
}
