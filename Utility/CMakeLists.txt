add_library(Utility INTERFACE
        vec3.h
        color.h
        ray.h
        hittable.h
        sphere.h rtweekend.h hittable_list.h camera.h)
target_include_directories(Utility INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})