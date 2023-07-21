#include "asteroid/renderer/renderer.h"
#include "asteroid/renderer/path_tracer.h"
#include "asteroid/util/macro.h"

using namespace Asteroid;

void Renderer::OnResize(unsigned int width, unsigned int height) {
    if (m_FinalImage) {
        // No resize necessary
        if (m_FinalImage->GetWidth() == width && m_FinalImage->GetHeight() == height)
            return;

        m_FinalImage->Resize(width, height);
    } else {
        m_FinalImage = std::make_shared<Image>(width, height);
    }

    auto pixel_num = width * height;

    m_devicePaths = std::make_unique<Buffer<PathSegment>>(pixel_num);

    m_Intersections = std::make_unique<Buffer<Intersection>>(pixel_num);

    m_AccumulationData = std::make_unique<Buffer<glm::vec3>>(pixel_num);

    m_ImageData = std::make_unique<Buffer<glm::u8vec4>>(pixel_num);

    ResetFrameIndex();
}

void Renderer::Render(const Scene &scene, const Camera &camera) {
    auto width = m_FinalImage->GetWidth();
    auto height = m_FinalImage->GetHeight();

    if (m_state.currentIteration == 0)
    {
        m_AccumulationData->clear();
    }

    m_state.currentIteration++;

    auto sceneView = SceneView(scene);
    auto paths = BufferView<PathSegment>(m_devicePaths->data(), m_devicePaths->size());
    auto its = BufferView<Intersection>(m_Intersections->data(), m_Intersections->size());
    auto accumulations = BufferView<glm::vec3>(m_AccumulationData->data(), m_AccumulationData->size());
    auto imageData = BufferView<glm::u8vec4>(m_ImageData->data(), m_ImageData->size());

    // Execute the kernel
    dim3 block(8, 8, 1);
    dim3 grid(width / block.x, height / block.y, 1);

    GeneratePathSegment<<<grid, block>>>(camera, m_state.traceDepth, paths);

    for (unsigned int i = 0; i < m_state.traceDepth; i++) {
        ComputeIntersection<<<grid, block>>>(sceneView, paths, width, height, its);

        Shading<<<grid, block>>>(sceneView, paths, its, width, height);
    }

    finalGather<<<grid, block>>>(accumulations, paths, width, height);

    ConvertToRGBA<<<grid, block>>>(accumulations, m_state.currentIteration, width, height, imageData);

    m_FinalImage->SetData(imageData.data());

}
