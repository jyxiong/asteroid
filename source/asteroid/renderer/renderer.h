#pragma once

#include <memory>
#include "glm/glm.hpp"
#include "asteroid/app/image.h"
#include "asteroid/renderer/scene.h"
#include "asteroid/renderer/scene_struct.h"

namespace Asteroid
{
class Renderer
{
public:
    void OnResize(int width, int height);

    void Render(const Scene& scene, const Camera& camera);

    std::shared_ptr<Image> GetFinalImage() const { return m_finalImage; }

    void ResetFrameIndex() { m_state.currentIteration = 0; }

    RenderState& GetRenderState() { return m_state; }

private:
    RenderState m_state;

    // 存储用于展示的纹理图像
    std::shared_ptr<Image> m_finalImage;

    std::unique_ptr<DeviceBuffer<glm::vec3>> m_AccumulationData = nullptr;

    // 存储最终颜色值[0, 255]
    std::unique_ptr<DeviceBuffer<glm::u8vec4>> m_ImageData = nullptr;

    std::unique_ptr<DeviceBuffer<PathSegment>> m_devicePaths = nullptr;

    std::unique_ptr<DeviceBuffer<Intersection>> m_Intersections = nullptr;
};
}
