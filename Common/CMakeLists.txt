add_library(Common STATIC
    aabb.h
    aabb.cpp
    camera.h
    camera.cpp
    color.h
    perlin.h
    perlin.cpp
    randomGenerator.h
    ray.h
    utils.h
    vec3.h
    vec3.cpp

    Material/dielectric.h
    Material/dielectric.cpp
    Material/diffuse_light.h
    Material/diffuse_light.cpp
    Material/isotropic.h
    Material/isotropic.cpp
    Material/lambertian.h
    Material/lambertian.cpp
    Material/metal.h
    Material/metal.cpp

    Object/aarect.h
    Object/aarect.cpp
    Object/box.h
    Object/box.cpp
    Object/bvh.h
    Object/bvh.cpp
    Object/constant_medium.h
    Object/constant_medium.cpp
    Object/hittable_list.h
    Object/hittable_list.cpp
    Object/moving_sphere.h
    Object/moving_sphere.cpp
    Object/sphere.h
    Object/sphere.cpp

    Texture/checker_texture.h
    Texture/checker_texture.cpp
    Texture/image_texture.h
    Texture/image_texture.cpp
    Texture/noise_texture.h
    Texture/noise_texture.cpp
    Texture/solid_color.h
    Texture/solid_color.cpp

    External/stb_image.h
    External/stb_image.c
    External/stb_image_write.h
    External/stb_image_write.c
    )


target_include_directories(Common PUBLIC ${CMAKE_SOURCE_DIR})