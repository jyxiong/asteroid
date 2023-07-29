#pragma once

#include <memory>
#include <cuda_runtime.h>

#include <optix.h>
#include <optix_stubs.h>

#include "glm/glm.hpp"
#include "asteroid/base/image.h"
#include "asteroid/renderer/scene.h"
#include "asteroid/renderer/scene_struct.h"
#include "asteroid/util/device_buffer.h"
#include "launchParams.h"

namespace Asteroid {
class Renderer {
public:
    Renderer();

    void OnResize(unsigned int width, unsigned int height);

    void Render();

    std::shared_ptr<Image> GetFinalImage() const { return m_finalImage; }

private:
    void initOptix();

    void createContext();

    void createModule();

    void createRaygenPG();

    void createMissPG();

    void createHitGroupPG();

    void createPipeline();

    void createSBT();

private:
    // 存储用于展示的纹理图像
    std::shared_ptr<Image> m_finalImage;
    DeviceBuffer m_colorBuffer;

    LaunchParams m_launchParams;
    DeviceBuffer m_launchParamsBuffer;

    CUcontext          m_cudaContext;
    CUstream           m_stream;
    cudaDeviceProp     m_deviceProps;

    OptixDeviceContext m_optixContext;

    OptixPipeline               m_pipeline;
    OptixPipelineCompileOptions m_pipelineCompileOptions = {};
    OptixPipelineLinkOptions    m_pipelineLinkOptions = {};

    OptixModule                 m_module;
    OptixModuleCompileOptions   m_moduleCompileOptions = {};

    std::vector<OptixProgramGroup> m_raygenPGs;
    DeviceBuffer m_raygenRecords;
    std::vector<OptixProgramGroup> m_missPGs;
    DeviceBuffer m_missRecords;
    std::vector<OptixProgramGroup> m_hitgroupPGs;
    DeviceBuffer m_hitgroupRecords;
    OptixShaderBindingTable m_sbt = {};

};
}
