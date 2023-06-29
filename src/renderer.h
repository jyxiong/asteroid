#include <string>

#include <glad/gl.h>
#include "GLFW/glfw3.h"

class Renderer
{
private:
    GLuint m_pbo = (GLuint) NULL;
    GLFWwindow *m_window;
    std::string m_yourName;
    unsigned int m_width;
    unsigned int m_height;
    int m_major;
    int m_minor;
    GLuint m_positionLocation = 0;
    GLuint m_texCoordsLocation = 1;
    GLuint m_image;

public:
    void initPBO(GLuint *pbo);
    void initTextures();
    void initVAO();
    GLuint initShader();

    void deletePBO(GLuint *pbo);
    void deleteTexture(GLuint *tex);
};