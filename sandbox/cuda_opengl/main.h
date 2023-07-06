#pragma once

#include <iostream>
#include <cstdlib>
#include <string>
#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include "kernel.h"

#include "asteroid/opengl/shader.h"
#include "asteroid/opengl/vertex_array.h"
#include "asteroid/opengl/texture2d.h"

using namespace Asteroid;

// ====================================
// GL stuff
// ====================================

GLuint             m_pbo = (GLuint) NULL;
GLFWwindow*    m_window;
std::string        m_yourName;
unsigned int       m_width;
unsigned int       m_height;
int                m_major;
int                m_minor;

cudaGraphicsResource_t m_resource;

std::shared_ptr<Shader> m_TextureShader;
std::shared_ptr<VertexArray> m_SquareVA;
std::shared_ptr<Texture2D> m_Texture;

// ====================================
// Main
// ====================================
int main(int argc, char* argv[]);

// ====================================
// Main loop
// ====================================
void mainLoop();
void runCUDA();

// ====================================
// Setup/init stuff
// ====================================
bool init(int argc, char **argv);
void initPBO();
void initCUDA();
void initTextures();
void initVAO();
void initShader();

// ====================================
// Clean-up stuff
// ====================================
void cleanupCUDA();
void deletePBO(GLuint *pbo);
void deleteTexture(GLuint *tex);
