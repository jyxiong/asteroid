#include "renderer.h"

#include "asteroid/util/macro.h"
#include "asteroid/shader/render.h"

using namespace Asteroid;

void renderFrame(const SceneView& scene,
                 const Camera& camera,
                 const RenderState& state,
                 BufferView<glm::vec4> image)
{
    dim3 block(16, 16, 1);
    dim3 grid(state.size.x / block.x, state.size.y / block.y, 1);

    renderFrameKernel<<<grid, block>>>(scene, camera, state, image);
}

Renderer::Renderer() {
    m_finalImage = std::make_shared<Image>();
}

void Renderer::onResize(const glm::ivec2& size)
{
    m_finalImage->onResize(size);
    m_state.size = size;
    m_imageData.resize(size.x * size.y);

    resetFrameIndex();
}

void Renderer::render(const Scene& scene, const Camera& camera)
{
    if (m_state.frame == 0)
    {
        m_imageData.clear();
    }

    auto sceneView = SceneView(scene);
    auto imageData = m_imageData.view();

    renderFrame(sceneView, camera, m_state, imageData);

    m_finalImage->setData(m_imageData.data());

    m_state.frame++;
}
