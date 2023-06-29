add_executable(main
    application.h
    main.cpp
    renderer.cpp
    renderer.h

    core/window.cpp
    core/window.h
    core/log.cpp
    core/log.h

    opengl/context.h
    opengl/shader.h
    opengl/texture.h)
target_link_libraries(main PUBLIC glad glfw glm spdlog)
