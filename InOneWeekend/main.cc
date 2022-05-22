#include <iostream>
#include <memory>
#include "Utility/vec3.h"
#include "Utility/ray.h"
#include "Utility/color.h"
#include "Utility/hittable.h"
#include "Utility/hittable_list.h"
#include "Utility/rtweekend.h"
#include "Utility/sphere.h"
#include "Utility/camera.h"
#include "Utility/material.h"

// blend (1.0, 1.0, 1.0) and (0.5, 0.7, 1.0) with height or ray.y()
color ray_color(const ray& r, const hittable_list& world, int depth)
{
    if (depth <= 0) // case 1: if not hit within depth recursive
        return color{0.0, 0.0, 0.0};

    hit_record rec;

    // case 0: hit the world geometry
    // set t_min to 0.001 rather than 0.0 due to the floating point approximation.
    if (world.hit(r, 0.001, infinity, rec)) // step 1: record the hit information of input ray and world geometry
    {
        ray scattered;
        color attenuation;

        if (rec.mat_ptr->scatter(r, rec, attenuation, scattered)) // step 2: get the scattered ray by record info
            return attenuation * ray_color(scattered, world, depth - 1); // step 3: blend the recursive color of the scattered ray
        return color{0.0, 0.0, 0.0}; // case 2: if not scatter
    }

    // case 3: hit the sky
    // cause viewport height is 2.0 in range (-1.0, 1.0)
    vec3 unit_direction = unit_vector(r.direction());
    auto t = 0.5 * (unit_direction.y() + 1.0);
    return (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
}

int main()
{
    // image
    const double aspect_ratio = 16.0 / 9.0;
    const int image_width = 400;
    const int image_height = static_cast<int>(image_width / aspect_ratio);
    const int samples_per_pixel = 100;
    const int depth = 50;

    // world
    hittable_list world;

    auto material_ground = std::make_shared<lambertian>(color(0.8, 0.8, 0.0));
    auto material_center = std::make_shared<lambertian>(color(0.1, 0.2, 0.5));
    auto material_left = std::make_shared<dielectric>(1.5);
    auto material_right = std::make_shared<metal>(color(0.8, 0.6, 0.2), 0.0);

    world.add(std::make_shared<sphere>(point3(0.0, -100.5, -1.0), 100.0, material_ground));
    world.add(std::make_shared<sphere>(point3(0.0, 0.0, -1.0), 0.5, material_center));
    world.add(std::make_shared<sphere>(point3(-1.0, 0.0, -1.0), 0.5, material_left));
    world.add(std::make_shared<sphere>(point3(-1.0, 0.0, -1.0), -0.45, material_left));
    world.add(std::make_shared<sphere>(point3(1.0, 0.0, -1.0), 0.5, material_right));

    // camera
    camera cam(point3(-2,2,1), point3(0,0,-1), vec3(0,1,0), 90, aspect_ratio);

    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    for (int j = image_height - 1; j >= 0; --j)
    {
        std::cerr << "\rScanline remaining: " << j << ' ' << std::flush;
        for (int i = 0; i < image_width; ++i)
        {
            color pixel_color{0.0, 0.0, 0.0};
            for (int k = 0; k < samples_per_pixel; ++k)
            {
                // ray
                auto u = (i + random_double()) / (image_width - 1);
                auto v = (j + random_double()) / (image_height - 1);

                ray ray = cam.get_ray(u, v);
                pixel_color += ray_color(ray, world, depth);

            }
            write_color(pixel_color, samples_per_pixel, std::cout);
        }
    }

    std::cerr << "\rDone.\n";

    return 0;
}