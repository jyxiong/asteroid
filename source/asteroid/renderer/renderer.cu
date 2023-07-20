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

    m_devicePaths = std::make_shared<DeviceBuffer<PathSegment>>(pixel_num);

    m_Intersections = std::make_shared<DeviceBuffer<Intersection>>(pixel_num);

    m_ImageData = std::make_shared<DeviceBuffer<glm::u8vec4>>(pixel_num);

}

void Renderer::Render(const Scene &scene, const Camera &camera) {
    auto width = m_FinalImage->GetWidth();
    auto height = m_FinalImage->GetHeight();

    auto sceneView = SceneView(scene);
    auto paths = DeviceBufferView<PathSegment>(*m_devicePaths);
    auto its = DeviceBufferView<Intersection>(*m_Intersections);
    auto imageData = DeviceBufferView<glm::u8vec4>(*m_ImageData);

    // Execute the kernel
    dim3 block(8, 8, 1);
    dim3 grid(width / block.x, height / block.y, 1);

    GeneratePrimaryRay<<<grid, block>>>(camera, paths);

    int bounces = 1;
    for (int i = 0; i < bounces; i++) {
        ComputeIntersection<<<grid, block>>>(sceneView, paths, width, height, its);

        Shading<<<grid, block>>>(sceneView, paths, its, width, height);
    }

    ConvertToRGBA<<<grid, block>>>(paths, width, height, imageData);

    m_FinalImage->SetData(imageData.data());
}
