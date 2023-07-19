#pragma once

#include <memory>
#include "glm/glm.hpp"
#include "asteroid/base/image.h"
#include "asteroid/renderer/camera.h"
#include "asteroid/renderer/ray.h"
#include "asteroid/renderer/intersection.h"
#include "asteroid/renderer/scene.h"
#include "asteroid/renderer/scene_struct.h"

namespace Asteroid
{
class Renderer
{
public:
    void OnResize(unsigned int width, unsigned int height);
    void Render(const Scene& scene, const Camera& camera);

    std::shared_ptr<Image> GetFinalImage() const { return m_FinalImage; }

private:
    // 存储用于展示的纹理图像
    std::shared_ptr<Image> m_FinalImage;

    const Scene* m_ActiveScene = nullptr;
    const Camera* m_ActiveCamera = nullptr;

    // 存储最终颜色值[0, 255]
    glm::u8vec4* m_ImageData = nullptr;

    std::shared_ptr<DeviceBuffer<PathSegment>> m_devicePaths = nullptr;

    Intersection* m_Intersections = nullptr;
};
}
