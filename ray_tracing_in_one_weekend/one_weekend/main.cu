#include <thrust/host_vector.h>

#include "glm/glm.hpp"

#include "color.h"
#include "camera.h"
#include "renderer.h"

int main()
{
    // 图像信息，单位是像素
    unsigned int image_width = 1600;
    unsigned int image_height = 900;

    // 相机外参
    auto position = glm::vec3(0);
    auto direction = glm::vec3(0, 0, -1);
    auto up = glm::vec3(0, 1, 1);
    Camera camera(position, direction,up);
    // 相机内参
    CameraParameters camera_params{};
    camera_params.fov = 120.f;
    camera_params.focal_length = 1.f;
    camera_params.viewport = { image_width, image_height };
    camera.set_parameters(camera_params);

    // framebuffer
    std::vector<glm::u8vec3> framebuffer(image_width * image_height);

    Renderer renderer(image_width, image_height);
    renderer.set_camera(camera);
    renderer.render(framebuffer);

    write_color(framebuffer, image_width, image_height);

    return 0;
}