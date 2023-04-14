#pragma once

#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
#include <thrust/device_vector.h>
#include "glm/glm.hpp"

__device__
glm::vec3 ray_color(const Ray &ray)
{
    auto unit_vector = glm::normalize(ray.direction);
//    auto t = 0.5 * (unit_vector.y + 1.f);
//    return glm::mix(glm::vec3(1.f), glm::vec3(0.5, 0.7, 1.0), t);
    return unit_vector;
}

class RenderFunctor
{
private:
    unsigned int m_width;
    unsigned int m_height;
    Camera *m_camera{ nullptr };
public:
    RenderFunctor(unsigned int image_width, unsigned int image_height)
            : m_width(image_width), m_height(image_height) {}

    void set_camera(Camera *camera)
    {
        m_camera = camera;
    }

    __device__
    glm::u8vec3 operator()(unsigned int pixel_index) const
    {
        auto idx = pixel_index % m_width;
        auto idy = pixel_index / m_width;

        Ray ray{};
        m_camera->generate_ray(idx, idy, ray);

        return ray_color(ray) * 255.f;
    }
};

class Renderer
{
private:
    unsigned int m_width;
    unsigned int m_height;
    unsigned int m_fb_size;
    thrust::host_vector<glm::u8vec3> m_framebuffer{};
    thrust::device_vector<glm::u8vec3> m_d_framebuffer{};

    std::shared_ptr<Camera> m_camera{ nullptr };
    thrust::device_ptr<Camera> m_d_camera{ nullptr };


public:
    Renderer(unsigned int width, unsigned int height)
            : m_width(width), m_height(height)
    {
        m_fb_size = m_width * m_height;
        m_framebuffer.resize(m_fb_size);
        m_d_framebuffer.resize(m_fb_size);

        m_d_camera = thrust::device_malloc<Camera>(1);
    }

    ~Renderer()
    {
        thrust::device_free(m_d_camera);
    }

    void set_camera(const std::shared_ptr<Camera> &camera)
    {
        m_camera = camera;
        cudaMemcpy(thrust::raw_pointer_cast(m_d_camera), m_camera.get(), sizeof(Camera), cudaMemcpyHostToDevice);
    }

    void render(glm::u8vec3 *framebuffer)
    {
        // 渲染函数
        RenderFunctor functor(m_width, m_height);
        functor.set_camera(m_d_camera.get());

        // 渲染
        thrust::counting_iterator<unsigned int> count_begin(0);
        thrust::counting_iterator<unsigned int> count_end(m_fb_size);
        thrust::transform(count_begin, count_end, m_d_framebuffer.begin(), functor);

        // 拷贝
        cudaMemcpy(framebuffer, thrust::raw_pointer_cast(m_d_framebuffer.data()), m_fb_size * sizeof(glm::u8vec3), cudaMemcpyDeviceToHost);
    }
};
