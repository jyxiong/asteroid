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
    Renderer();

    void onResize(const glm::ivec2& resolution);

    void render(const Scene& scene, const Camera& camera);

    std::shared_ptr<Image> getFinalImage() const { return m_finalImage; }

    void resetFrameIndex() { m_state.frame = 0; }

    RenderState& getRenderState() { return m_state; }

private:
    RenderState m_state;

    // 存储用于展示的纹理图像
    std::shared_ptr<Image> m_finalImage;

    DeviceBuffer<glm::vec3> m_accumulationData;

    // 存储最终颜色值[0, 255]
    DeviceBuffer<glm::vec4> m_imageData;

    DeviceBuffer<PathSegment> m_devicePaths;

    DeviceBuffer<Intersection> m_intersections;
};
}
