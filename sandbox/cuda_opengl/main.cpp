#include <cstdio>
#include <sstream>
#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include "main.h"

/**
 * C main function.
 */
int main(int argc, char* argv[]) {
    // TODO: Change this line to use your name!
    m_yourName = "TODO: YOUR NAME HERE";

    if (init(argc, argv)) {
        mainLoop();
    }

    return 0;
}

/**
 * Initialization of CUDA and GLFW.
 */
bool init(int argc, char **argv) {
    // Set window title to "Student Name: [SM 2.0] GPU Name"
    std::string deviceName;
    cudaDeviceProp deviceProp;
    int gpuDevice = 0;
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (gpuDevice > device_count) {
        std::cout << "Error: GPU device number is greater than the number of devices!" <<
                  "Perhaps a CUDA-capable GPU is not installed?" << std::endl;
        return false;
    }
    cudaGetDeviceProperties(&deviceProp, gpuDevice);
    m_major = deviceProp.major;
    m_minor = deviceProp.minor;

    std::ostringstream ss;
    ss << m_yourName << ": [SM " << m_major << "." << m_minor << "] " << deviceProp.name;
    deviceName = ss.str();

    // Window setup stuff

    if (!glfwInit()) {
        return false;
    }
    m_width = 800;
    m_height = 800;
    m_window = glfwCreateWindow(m_width, m_height, deviceName.c_str(), NULL, NULL);
    if (!m_window) {
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(m_window);

    auto status = gladLoadGL(glfwGetProcAddress);
    if (!status) {
        return false;
    }

    // init all of the things
    initVAO();
    initTextures();
    initCUDA();
    initPBO();

    initShader();


    return true;
}

void initPBO() {
        int num_texels = m_width * m_height;
        int num_values = num_texels * 4;
        size_t size_tex_data = sizeof(GLubyte) * num_values;

        // Generate a buffer ID called a PBO (Pixel Buffer Object)
        glGenBuffers(1, &m_pbo);
        // Make this the current UNPACK buffer (OpenGL is state-based)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
        // Allocate data for the buffer. 4-channel 8-bit image
        glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
        // ע�ᵽcuda
        cudaGraphicsGLRegisterBuffer(&m_resource, m_pbo, cudaGraphicsMapFlagsNone);
}

void initVAO() {
    m_SquareVA = std::make_shared<VertexArray>();

    float squareVertices[5 * 4] = {
        -1.f, -1.f, 0.f, 1.f, 1.f,
        1.f, -1.f, 0.f, 0.f, 1.f,
        1.f, 1.f, 0.f, 0.f, 0.f,
        -1.f, 1.f, 0.f, 1.f, 0.f
    };

    auto squareVB = std::make_shared<VertexBuffer>(squareVertices, sizeof(squareVertices));
    squareVB->SetLayout({ { ShaderDataType::Float3, "a_Position" },
                         { ShaderDataType::Float2, "a_TexCoord" } });
    m_SquareVA->AddVertexBuffer(squareVB);

    unsigned int squareIndices[6] = { 0, 1, 3, 3, 1, 2 };
    auto squareIB = std::make_shared<IndexBuffer>(squareIndices, sizeof(squareIndices) / sizeof(unsigned int));
    m_SquareVA->SetIndexBuffer(squareIB);
}

void initCUDA() {
    // Default to device ID 0. If you have more than one GPU and want to test a non-default one,
    // change the device ID.
    cudaGLSetGLDevice(0);

    // Clean up on program exit
    atexit(cleanupCUDA);
}

void initTextures() {
    
    TextureSpecification texSpec{};
    texSpec.Width = m_width;
    texSpec.Height = m_height;
    texSpec.Format = InternalFormat::RGBA8;
    texSpec.pixel_format = PixelFormat::RGBA;
    texSpec.GenerateMips = false;
    m_Texture = std::make_shared<Texture2D>(texSpec);
}

void initShader() {
    std::string textureShaderVertexSrc = R"(
			#version 330 core

            layout(location = 0) in vec3 a_Position;
			layout(location = 1) in vec2 a_TexCoord;
			out vec2 v_TexCoord;
			void main()
			{
				v_TexCoord = a_TexCoord;
				gl_Position = vec4(a_Position, 1.0);
			}
		)";

    std::string textureShaderFragmentSrc = R"(
			#version 330 core

            layout(location = 0) out vec4 color;
			in vec2 v_TexCoord;

			uniform sampler2D u_Texture;
			void main()
			{
				color = texture(u_Texture, v_TexCoord);
			}
		)";

    m_TextureShader = std::make_shared<Shader>(textureShaderVertexSrc, textureShaderFragmentSrc);
}

// ====================================
// Main loop stuff
// ====================================

void runCUDA() {

    size_t num_bytes;

    uchar4 *dptr = NULL;
    cudaGraphicsMapResources(1, &m_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes, m_resource);

    // Execute the kernel
    kernelVersionVis(dptr, m_width, m_height, m_major, m_minor);

    // Unmap buffer object
    cudaGraphicsUnmapResources(1, &m_resource, 0);
}

void mainLoop() {
    while (!glfwWindowShouldClose(m_window)) {
        glfwPollEvents();
        runCUDA();

        // https://www.cnblogs.com/crsky/p/7870835.html
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
        m_Texture->Bind();
        m_Texture->SetData(nullptr, 0);
        //glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_width, m_height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glClear(GL_COLOR_BUFFER_BIT);

        // VAO, shader program, and texture already bound
        m_SquareVA->Bind();
        m_TextureShader->Bind();
        glDrawElements(GL_TRIANGLES, m_SquareVA->GetIndexBuffer()->GetCount(), GL_UNSIGNED_INT, 0);
        glfwSwapBuffers(m_window);
    }
    glfwDestroyWindow(m_window);
    glfwTerminate();
}

// ====================================
// Clean-up stuff
// ====================================

void cleanupCUDA() {
    if (m_pbo) {
        deletePBO(&m_pbo);
    }
}

void deletePBO(GLuint *pbo) {
    if (pbo) {
        // unregister this buffer object with CUDA
        cudaGraphicsUnregisterResource(m_resource);

        glBindBuffer(GL_ARRAY_BUFFER, *pbo);
        glDeleteBuffers(1, pbo);

        *pbo = (GLuint)NULL;
    }
}

void deleteTexture(GLuint *tex) {
    glDeleteTextures(1, tex);
    *tex = (GLuint)NULL;
}
