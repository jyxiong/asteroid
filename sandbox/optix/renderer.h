#pragma once

#include <memory>
#include <cuda_runtime.h>

#include <optix.h>
#include <optix_stubs.h>

#include "glm/glm.hpp"
#include "asteroid/app/image.h"
#include "asteroid/renderer/scene.h"
#include "asteroid/renderer/scene_struct.h"
#include "asteroid/util/device_buffer.h"
#include "asteroid/util/matrix.h"
#include "launchParams.h"

namespace Asteroid {

struct TriangleMesh {
    /*! add a unit cube (subject to given xfm matrix) to the current
        triangleMesh */
    void addUnitCube(const glm::mat4 &xfm);

    //! add aligned cube aith front-lower-left corner and size
    void addCube(const glm::vec3 &center, const glm::vec3 &size);

    std::vector<glm::vec3> vertex;
    std::vector<glm::ivec3> index;
    glm::vec3              color;

};

class Renderer {
public:
    Renderer();

    void OnResize(unsigned int width, unsigned int height);

    void Render();

    std::shared_ptr<Image> GetFinalImage() const { return m_finalImage; }

    void setModel(const std::vector<TriangleMesh> &model);

    void setCamera(const Camera &camera);

private:
    void initOptix();

    void createContext();

    void createModule();

    void createRaygenPG();

    void createMissPG();

    void createHitGroupPG();

    void createPipeline();

    void createSBT();

    void createAccel();

private:
    // 存储用于展示的纹理图像
    std::shared_ptr<Image> m_finalImage;
    DeviceBuffer m_colorBuffer;

    Camera m_camera;

    std::vector<TriangleMesh> m_meshes;
    std::vector<DeviceBuffer> m_vertexBuffer;
    std::vector<DeviceBuffer> m_indexBuffer;
    DeviceBuffer m_asBuffer;

    LaunchParams m_launchParams;
    DeviceBuffer m_launchParamsBuffer;

    CUcontext m_cudaContext{};
    CUstream m_stream{};
    cudaDeviceProp m_deviceProps{};

    OptixDeviceContext m_optixContext{};

    OptixPipeline m_pipeline{};
    OptixPipelineCompileOptions m_pipelineCompileOptions = {};
    OptixPipelineLinkOptions m_pipelineLinkOptions = {};

    OptixModule m_module{};
    OptixModuleCompileOptions m_moduleCompileOptions = {};

    std::vector<OptixProgramGroup> m_raygenPGs;
    DeviceBuffer m_raygenRecords;
    std::vector<OptixProgramGroup> m_missPGs;
    DeviceBuffer m_missRecords;
    std::vector<OptixProgramGroup> m_hitgroupPGs;
    DeviceBuffer m_hitgroupRecords;
    OptixShaderBindingTable m_sbt = {};

};
}
