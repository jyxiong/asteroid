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

// blend (1.0, 1.0, 1.0) and (0.5, 0.7, 1.0) with height or ray.y()
color ray_color(const ray& r, const hittable_list& world)
{
    hit_record rec;

    if (world.hit(r, 0.0, infinity, rec))
        return 0.5 * (rec.normal + 1.0);

    // cause viewport height is 2.0 in range (-1.0, 1.0)
    auto t = 0.5 * (r.direction().y() + 1.0);
    return (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
}


int main()
{
    // image
    const double aspect_ratio = 16.0 / 9.0;
    const int image_width = 400;
    const int image_height = static_cast<int>(image_width / aspect_ratio);
    const int samples_per_pixel = 100;

    // world
    hittable_list world;
    world.add(std::make_shared<sphere>(point3(0.0, 0.0, -1.0), 0.5));
    world.add(std::make_shared<sphere>(point3(0.0, -100.5, -1.0), 100));

    // camera
    camera cam;

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
                pixel_color += ray_color(ray, world);

            }
            write_color(pixel_color, samples_per_pixel, std::cout);
        }
    }

    std::cerr << "\rDone.\n";

    return 0;
}