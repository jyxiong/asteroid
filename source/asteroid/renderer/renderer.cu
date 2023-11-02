#include "renderer.h"

#include "asteroid/util/macro.h"
#include "path_tracer.h"

using namespace Asteroid;

Renderer::Renderer() {
    m_finalImage = std::make_shared<Image>();
}

void Renderer::onResize(const glm::ivec2& size)
{
    if (m_state.size == size)
        return;

    m_finalImage->resize(size);
    m_state.size = size;

    auto pixel_num = size.x * size.y;

    m_devicePaths.resize(pixel_num);
    m_intersections.resize(pixel_num);
    m_imageData.resize(pixel_num);

    resetFrameIndex();
}

void Renderer::render(const Scene& scene, const Camera& camera)
{
    if (m_state.frame == 0)
    {
        m_imageData.clear();
    }

    auto sceneView = SceneView(scene);
    auto paths = m_devicePaths.view();
    auto its = m_intersections.view();
    auto imageData = m_imageData.view();

    // Execute the kernel
    dim3 block(8, 8, 1);
    dim3 grid(m_state.size.x / block.x, m_state.size.y / block.y, 1);

    generatePathSegment<<<grid, block>>>(camera, m_state.maxDepth, paths);

    for (unsigned int i = 0; i < m_state.maxDepth; i++)
    {
        computeIntersection<<<grid, block>>>(sceneView, m_state.size.x, m_state.size.y, paths, its);

        shading<<<grid, block>>>(sceneView, its, m_state.size, paths);
    }

    finalGather<<<grid, block>>>(paths, m_state.frame, m_state.size.x, m_state.size.y, imageData);

    m_finalImage->setData(m_imageData.data());

    m_state.frame++;
}
