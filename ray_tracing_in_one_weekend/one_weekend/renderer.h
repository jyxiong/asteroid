#pragma once

#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
#include <thrust/device_vector.h>
#include "glm/glm.hpp"

__device__
glm::vec3 ray_color(const Ray &ray)
{
    auto unit_vector = glm::normalize(ray.direction);
    auto t = 0.5 * (unit_vector.y + 1.f);
    return glm::mix(glm::vec3(1.f), glm::vec3(0.5, 0.7, 1.0), t);
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

        auto ray = m_camera->generate_ray(idx, idy);

        return ray_color(ray) * 255.f;
    }
};

class Renderer
{
private:
    unsigned int m_width;
    unsigned int m_height;
    unsigned int m_fb_size;

    thrust::device_vector<glm::u8vec3> m_d_framebuffer{};
    thrust::device_vector<Camera> m_d_camera{ 1 };

public:
    Renderer(unsigned int width, unsigned int height)
        : m_width(width), m_height(height)
    {
        m_fb_size = m_width * m_height;
        m_d_framebuffer.resize(m_fb_size);
    }

    ~Renderer() = default;

    void set_camera(const Camera &camera)
    {
        thrust::copy_n(&camera, 1, m_d_camera.begin());
    }

    void render(std::vector<glm::u8vec3> &framebuffer)
    {
        RenderFunctor functor(m_width, m_height);
        functor.set_camera(m_d_camera.data().get());

        thrust::counting_iterator<unsigned int> count_begin(0);
        thrust::counting_iterator<unsigned int> count_end(m_fb_size);
        thrust::transform(count_begin, count_end, m_d_framebuffer.begin(), functor);

        thrust::copy(m_d_framebuffer.begin(), m_d_framebuffer.end(), framebuffer.begin());
    }
};
