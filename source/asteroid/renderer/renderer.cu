#include "renderer.h"

#include "asteroid/util/macro.h"
#include "path_tracer.h"

using namespace Asteroid;

void Renderer::OnResize(unsigned int width, unsigned int height) {
    if (m_finalImage) {
        // No resize necessary
        if (m_finalImage->GetWidth() == width && m_finalImage->GetHeight() == height)
            return;

        m_finalImage->Resize(width, height);
    } else {
        m_finalImage = std::make_shared<Image>(width, height);
    }

    auto pixel_num = width * height;

    m_devicePaths = std::make_unique<DeviceBuffer<PathSegment>>(pixel_num);

    m_Intersections = std::make_unique<DeviceBuffer<Intersection>>(pixel_num);

    m_AccumulationData = std::make_unique<DeviceBuffer<glm::vec3>>(pixel_num);

    m_ImageData = std::make_unique<DeviceBuffer<glm::u8vec4>>(pixel_num);

    ResetFrameIndex();
}

void Renderer::Render(const Scene &scene, const Camera &camera) {
    auto width = m_finalImage->GetWidth();
    auto height = m_finalImage->GetHeight();

    if (m_state.currentIteration == 0)
    {
        m_AccumulationData->clear();
    }

    m_state.currentIteration++;

    auto sceneView = SceneView(scene);
    auto paths = m_devicePaths->view();
    auto its = m_Intersections->view();
    auto accumulations = m_AccumulationData->view();
    auto imageData = m_ImageData->view();

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

    m_finalImage->SetData(m_ImageData->data());

}
