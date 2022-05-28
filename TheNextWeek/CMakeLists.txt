add_executable(TheNextWeek
        main.cc
        hittable.h
        hittable_list.h
        sphere.h
        material.h
        bvh.h
        moving_sphere.h
        )

target_include_directories(TheNextWeek PUBLIC
        ${CMAKE_SOURCE_DIR})

target_link_libraries(TheNextWeek PUBLIC
        Common
        )