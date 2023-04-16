#include <vector>

#include "glm/glm.hpp"

#include "color.h"
#include "camera.h"
#include "renderer.h"
#include "sphere.h"
#include "hittable.h"
#include "hittable_list.h"

__global__ void create_world(Hittable **list, Hittable **world)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        list[0] = new Sphere(glm::vec3(0, 0, -1), 0.5);
        list[1] = new Sphere(glm::vec3(0, -100.5, -1), 100);
        *world = new HittableList(list, 2);
    }
}

int main()
{
    // 图像信息，单位是像素
    unsigned int width = 1600;
    unsigned int height = 900;

    // 相机外参
    auto position = glm::vec3(0);
    auto direction = glm::vec3(0, 0, -1);
    auto up = glm::vec3(0, 1, 0);
    Camera camera(position, direction, up);
    // 相机内参
    CameraParameters camera_params{};
    camera_params.fov = 120.f;
    camera_params.focal_length = 1.f;
    camera_params.film_size = { width, height };
    camera.set_parameters(camera_params);

    Camera *d_camera;
    checkCudaErrors(cudaMalloc(&d_camera, sizeof(Camera)));
    checkCudaErrors(cudaMemcpy(d_camera, &camera, sizeof(Camera), cudaMemcpyHostToDevice));

    Hittable **d_list, **d_world;
    checkCudaErrors(cudaMalloc((void **) &d_list, 2 * sizeof(Hittable *)));
    checkCudaErrors(cudaMalloc((void **) &d_world, sizeof(Hittable *)));
    create_world<<<1, 1>>>(d_list, d_world);

    glm::u8vec3 *d_fb;
    checkCudaErrors(cudaMalloc(&d_fb, width * height * sizeof(glm::u8vec3)));

    unsigned int block_width = 32, block_height = 32;
    dim3 block_size(block_width, block_height);
    dim3 grid_size(width / block_height + 1, height / block_height + 1);
    render<<<grid_size, block_size>>>(width, height, d_camera, d_world, d_fb);

    std::vector<glm::u8vec3> fb(width * height);
    checkCudaErrors(cudaMemcpy(fb.data(), d_fb, width * height * sizeof(glm::u8vec3), cudaMemcpyDeviceToHost));

    write_color(fb, width, height);

    return 0;
}