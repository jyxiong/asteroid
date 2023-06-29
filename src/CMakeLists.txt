add_executable(main
        main.cpp
        renderer.cpp
        renderer.h

        opengl/shader.h
        opengl/texture.h)
target_link_libraries(main PUBLIC glad glfw glm)
