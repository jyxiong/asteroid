add_executable(InOneWeekend
        main.cc
        hittable.h
        hittable_list.h
        sphere.h
        material.h
        )

target_include_directories(InOneWeekend PUBLIC
        ${CMAKE_SOURCE_DIR})

target_link_libraries(InOneWeekend PUBLIC
        Common
        )