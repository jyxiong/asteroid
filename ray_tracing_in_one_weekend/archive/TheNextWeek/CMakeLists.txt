add_executable(TheNextWeek
        main.cc
        ../Common/Object/hittable.h
        ../Common/Object/hittable_list.h
        ../Common/Object/sphere.h
        ../Common/Material/material.h
        ../Common/Object/bvh.h
        ../Common/Object/moving_sphere.h
        )

target_include_directories(TheNextWeek PUBLIC
        ${CMAKE_SOURCE_DIR})

target_link_libraries(TheNextWeek PUBLIC
        Common
        )