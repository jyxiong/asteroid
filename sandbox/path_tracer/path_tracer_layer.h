#include <memory>
#include "cuda_runtime.h"
#include "asteroid/base/layer.h"
#include "asteroid/base/image.h"
#include "asteroid/opengl/shader.h"
#include "asteroid/opengl/vertex_array.h"
#include "asteroid/opengl/texture2d.h"

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
    void InitImage();

    void InitShader();

    void InitVao();

    void InitCuda();

    void Render();

    void Preview();

private:
    
    std::shared_ptr<Image> m_Image;
    
    std::shared_ptr<Shader> m_Shader;
    
    std::shared_ptr<VertexArray> m_Vao;

    uchar4* m_ImageData = nullptr;
};

}