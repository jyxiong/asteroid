#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <glm/glm.hpp>

#include "color.h"
#include "ray.h"

class RenderFunctor
{
private:
    unsigned int m_image_width;
    unsigned int m_image_height;

public:
    RenderFunctor(unsigned int image_width, unsigned int image_height)
            : m_image_width(image_width), m_image_height(image_height)
    {}

    __device__
    glm::u8vec3 operator()(unsigned int pixel_index) const
    {
        auto idx = pixel_index % m_image_width;
        auto idy = pixel_index / m_image_width;

        auto r = (float)idx / (float)m_image_width;
        auto g = (float)idy / (float)m_image_height;
        auto b = 0.25f;

        return { r * 255, g * 255, b * 255 };
    }
};

int main()
{
    unsigned int image_width = 1280;
    unsigned int image_height = 960;

    size_t framebuffer_size = image_width * image_height;
    thrust::device_vector<glm::u8vec3> d_framebuffer(framebuffer_size);

    RenderFunctor functor(image_width, image_height);

    thrust::counting_iterator<unsigned int> count_begin(0);
    thrust::counting_iterator<unsigned int> count_end(framebuffer_size);
    thrust::transform(count_begin, count_end, d_framebuffer.begin(), functor);

    thrust::host_vector<glm::u8vec3> h_framebuffer(framebuffer_size);
    thrust::copy(d_framebuffer.begin(),d_framebuffer.end(), h_framebuffer.begin());

    write_color(h_framebuffer, image_width, image_height);

    return 0;
}