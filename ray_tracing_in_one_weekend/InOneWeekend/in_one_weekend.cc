#include <iostream>
#include <memory>

#include "Common/External/stb_image_write.h"

#include "Common/Object/sphere.h"
#include "Common/Object/hittable.h"
#include "Common/Object/hittable_list.h"

#include "Common/vec3.h"
#include "Common/ray.h"
#include "Common/color.h"
#include "Common/rng.h"
#include "Common/camera.h"

#include "Common/Material/lambertian.h"
#include "Common/Material/metal.h"
#include "Common/Material/dielectric.h"

color ray_color(const ray &r, const hittable_list &world, int depth) {
    if (depth <= 0) // 第一种情况：深度终止
        return {0.0, 0.0, 0.0};

    hit_record rec;

    // 第二种情况：与世界中的物体发生碰撞
    // t_min设置为0.001，避免浮点数精度问题导致的shadow ance
    if (world.hit(r, 0.001, util::infinity, rec)) // 第一步：碰撞检测，计算入射光线与世界中的物体的碰撞信息rec
    {
        ray scattered;
        color attenuation;

        if (rec.mat_ptr->scatter(r, rec, attenuation, scattered)) // 第二步：散射着色，根据碰撞信息计算散射光线和颜色
            return attenuation
                * ray_color(scattered, world, depth - 1); // 第三步：递归对散射光线进行光线追踪

        return {0.0, 0.0, 0.0}; // 第二步：不散射光线
    }

    // 第三种情况：与天空发生碰撞
    vec3 unit_direction = unit_vector(r.direction());
    auto t = 0.5 * (unit_direction.y() + 1.0); // y[-1, 1] ——> t[0, 1]
    return (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0); // 根据t将天空色(0.5, 0.7, 1.0)与白色混合
}

hittable_list random_scene() {
    hittable_list world;

    auto ground_material = std::make_shared<lambertian>(color(0.5, 0.5, 0.5));
    world.add(std::make_shared<sphere>(point3(0, -1000, 0), 1000, ground_material));

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            auto choose_mat = random_double();
            point3 center(a + 0.9 * random_double(), 0.2, b + 0.9 * random_double());

            if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                std::shared_ptr<material> sphere_material;

                if (choose_mat < 0.8) {
                    // diffuse
                    auto albedo = random_vec3() * random_vec3();
                    sphere_material = std::make_shared<lambertian>(albedo);
                    world.add(std::make_shared<sphere>(center, 0.2, sphere_material));
                } else if (choose_mat < 0.95) {
                    // metal
                    auto albedo = random_vec3(0.5, 1);
                    auto fuzz = random_double(0, 0.5);
                    sphere_material = std::make_shared<metal>(albedo, fuzz);
                    world.add(std::make_shared<sphere>(center, 0.2, sphere_material));
                } else {
                    // glass
                    sphere_material = std::make_shared<dielectric>(1.5);
                    world.add(std::make_shared<sphere>(center, 0.2, sphere_material));
                }
            }
        }
    }

    auto material1 = std::make_shared<dielectric>(1.5);
    world.add(std::make_shared<sphere>(point3(0, 1, 0), 1.0, material1));

    auto material2 = std::make_shared<lambertian>(color(0.4, 0.2, 0.1));
    world.add(std::make_shared<sphere>(point3(-4, 1, 0), 1.0, material2));

    auto material3 = std::make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
    world.add(std::make_shared<sphere>(point3(4, 1, 0), 1.0, material3));

    return world;
}

int main() {
    // image
    const double aspect_ratio = 16.0 / 9.0;
    const int image_width = 400;
    const int image_height = static_cast<int>(image_width / aspect_ratio);
    const int samples_per_pixel = 100;
    const int depth = 50;

    // world
    auto world = random_scene();

    // camera
    point3 lookfrom(13, 2, 3);
    point3 lookat(0, 0, 0);
    vec3 vup(0, 1, 0);
    auto dist_to_focus = 10.0;
    auto aperture = 0.1;

    camera cam(lookfrom, lookat, vup, 20, aspect_ratio, aperture, dist_to_focus);

    std::vector<unsigned char> image(image_width * image_height * 3);
    for (int j = 0; j < image_height; ++j) {
        std::cerr << "\rScanline remaining: " << j << ' ' << std::flush;
        for (int i = 0; i < image_width; ++i) {
            color pixel_color{0.0, 0.0, 0.0};
            for (int k = 0; k < samples_per_pixel; ++k) {
                // ray
                auto u = (i + random_double()) / (image_width - 1);
                auto v = (j + random_double()) / (image_height - 1);

                ray ray = cam.get_ray(u, v);
                pixel_color += ray_color(ray, world, depth);
            }
            write_color(pixel_color, samples_per_pixel, j * image_width + i, image);
        }
    }

    stbi_flip_vertically_on_write(true);
    stbi_write_jpg("InOneWeekend1.jpg", image_width, image_height, 3, image.data(), 100);

    std::cerr << "\rDone.\n";

    return 0;
}