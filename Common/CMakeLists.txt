add_library(Common INTERFACE
        vec3.h
        color.h
        ray.h
        rtweekend.h
        camera.h
        texture.h
        aabb.h
        rtw_stb_image.h
        )
target_include_directories(Common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})