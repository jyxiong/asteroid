#pragma once

#include <memory>
#include "glm/glm.hpp"
#include "asteroid/base/image.h"
#include "asteroid/renderer/camera.h"
#include "asteroid/renderer/ray.h"

namespace Asteroid
{
class Renderer
{
public:
    void OnResize(unsigned int width, unsigned int height);
    void Render(const Camera& camera);

    std::shared_ptr<Image> GetFinalImage() const { return m_FinalImage; }

private:
    // 存储计算过程中颜色值[0, 1]
    glm::vec4* m_AccumulationData = nullptr;

    // 存储最终颜色值[0, 255]
    glm::u8vec4* m_ImageData = nullptr;

    Ray* m_Ray = nullptr;

    // 存储用于展示的纹理图像
    std::shared_ptr<Image> m_FinalImage;
};
}
