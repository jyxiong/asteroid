add_executable(InOneWeekend
    main.cc
    )

target_include_directories(InOneWeekend PUBLIC
    ${CMAKE_SOURCE_DIR})

target_link_libraries(InOneWeekend PUBLIC
    Common
    )