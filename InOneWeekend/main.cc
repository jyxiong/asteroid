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
color ray_color(const ray& r, const hittable_list& world, int depth)
{
    if (depth <= 0)
        return color{0.0, 0.0, 0.0};

    hit_record rec;

    // set t_min to 0.001 rather than 0.0 due to the floating point approximation.
    if (world.hit(r, 0.001, infinity, rec))
    {
        // https://raytracing.github.io/images/fig-1.09-rand-vec.jpg
        // here generate a random point in a unit sphere targeting the sphere at point p
        // reflect light is p->s

        // auto target = rec.p + rec.normal + random_in_unit_sphere();
        auto target = rec.p + rec.normal + random_unit_vector();
        // auto target = rec.p + random_in_hemisphere(rec.normal);

        // recursive to simulate the multiple reflection
        // what does 0.5 mean? may mean the light absorption of each reflection
        return 0.5 * ray_color(ray(rec.p, target - rec.p), world, depth - 1);
    }

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
                pixel_color += ray_color(ray, world, depth);

            }
            write_color(pixel_color, samples_per_pixel, std::cout);
        }
    }

    std::cerr << "\rDone.\n";

    return 0;
}