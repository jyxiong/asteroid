#include <memory>
#include "cuda_runtime.h"
#include "asteroid/base/layer.h"
#include "asteroid/base/image.h"
#include "asteroid/opengl/shader.h"
#include "asteroid/opengl/vertex_array.h"
#include "asteroid/opengl/texture2d.h"
#include "asteroid/opengl/framebuffer.h"

namespace Asteroid
{
class ExampleLayer : public Layer
{
public:
    ExampleLayer();

    ~ExampleLayer();

    void OnUpdate() override;

    void OnImGuiRender() override;

    void OnEvent(Event &event) override;

private:
    void InitCuda();
    void InitImage();

    void InitShader();
    void InitVao();
    void InitFbo();

    void Render();

    void Preview();

private:
    uchar4* m_ImageData = nullptr;
    std::shared_ptr<Image> m_Image;
    std::shared_ptr<Texture2D> m_Texture;
    std::shared_ptr<Shader> m_Shader;
    std::shared_ptr<VertexArray> m_Vao;

    std::shared_ptr<Framebuffer> m_Fbo;
};

}