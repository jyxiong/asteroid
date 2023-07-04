#include <memory>
#include "cuda_runtime.h"
#include "asteroid/base/layer.h"
#include "asteroid/opengl/shader.h"
#include "asteroid/opengl/vertex_array.h"
#include "asteroid/opengl/texture2d.h"

namespace Asteroid
{
class ExampleLayer : public Layer
{
public:
    ExampleLayer();

    void OnUpdate() override;

    void OnImGuiRender() override;

    void OnEvent(Event &event) override;

private:
    void InitShader();

    void InitVao();

    void InitTexture();

    void InitCuda();

    void UpdateCuda();

    void UpdateOpengl();

private:
    std::shared_ptr<Shader> m_TextureShader;
    std::shared_ptr<VertexArray> m_SquareVA;
    std::shared_ptr<Texture2D> m_Texture;

    cudaGraphicsResource_t m_CudaResource;
    unsigned char *m_Data;
};

}