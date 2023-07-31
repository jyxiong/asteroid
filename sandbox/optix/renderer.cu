#include "renderer.h"

// this include may only appear in a single source file:
#include <optix_function_table_definition.h>

#include "asteroid/util/log.h"
#include "otkHelloKernelPTX.h"

using namespace Asteroid;

//extern "C" char embedded_ptx_code[];

static void context_log_cb(unsigned int level,
                           const char *tag,
                           const char *message,
                           void *)
{
    fprintf(stderr, "[%2d][%12s]: %s\n", (int) level, tag, message);
}

/*! SBT record for a raygen program */
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // just a dummy value - later examples will use more interesting
    // data here
    void *data;
};

/*! SBT record for a miss program */
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // just a dummy value - later examples will use more interesting
    // data here
    void *data;
};

/*! SBT record for a hitgroup program */
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // just a dummy value - later examples will use more interesting
    // data here
    int objectID;
};

Renderer::Renderer()
{
    initOptix();

    AST_CORE_INFO("creating optix context ...");
    createContext();

    AST_CORE_INFO("creating optix module ...");
    createModule();

    AST_CORE_INFO("create optix raygen program ...");
    createRaygenPG();

    AST_CORE_INFO("creating optix miss program ...");
    createMissPG();

    AST_CORE_INFO("creating optix hit group program ...");
    createHitGroupPG();

    AST_CORE_INFO("creating optix pipeline ...");
    createPipeline();

    AST_CORE_INFO("creating optix shader binding table ...");
    createSBT();

    m_launchParamsBuffer.alloc(sizeof(m_launchParams));
}

void Renderer::OnResize(unsigned int width, unsigned int height)
{
    if (m_finalImage)
    {
        // No resize necessary
        if (m_finalImage->GetWidth() == width && m_finalImage->GetHeight() == height)
            return;

        m_finalImage->Resize(width, height);
    } else
    {
        m_finalImage = std::make_shared<Image>(width, height);
    }

    m_colorBuffer.resize(width * height * sizeof(unsigned int));

    m_launchParams.width = width;
    m_launchParams.height = height;
    m_launchParams.colorBuffer = (unsigned int *) m_colorBuffer.devicePtr();
}

void Renderer::Render()
{
    if (m_launchParams.width == 0) return;

    m_launchParamsBuffer.upload(&m_launchParams,1);
    m_launchParams.frameID++;

    AST_OPTIX_CHECK(optixLaunch(/*! pipeline we're launching launch: */
        m_pipeline,m_stream,
        /*! parameters and SBT */
        (CUdeviceptr)m_launchParamsBuffer.devicePtr(),
        m_launchParamsBuffer.m_sizeInBytes,
        &m_sbt,
        /*! dimensions of the launch: */
        m_launchParams.width,
        m_launchParams.height,
        1
    ));
    // sync - make sure the frame is rendered before we download and
    // display (obviously, for a high-performance application you
    // want to use streams and double-buffering, but for this simple
    // example, this will have to do)
    AST_CUDA_SYNC_CHECK();

    m_finalImage->SetData(m_colorBuffer.devicePtr());
}

void Renderer::initOptix()
{
    AST_CORE_INFO("init optix ...");

    cudaFree(nullptr);
    int numDevices;
    cudaGetDeviceCount(&numDevices);
    if (numDevices == 0)
    {
        AST_CORE_ERROR("no CUDA capable devices found!");
    }

    AST_CORE_INFO("found {0} CUDA capable devices", numDevices);

    AST_OPTIX_CHECK(optixInit());

    AST_CORE_INFO("initialize optix successfully!");
}

void Renderer::createContext()
{
    const int deviceID = 0;
    AST_CUDA_CHECK(cudaSetDevice(deviceID));
    AST_CUDA_CHECK(cudaStreamCreate(&m_stream));

    cudaGetDeviceProperties(&m_deviceProps, deviceID);
    AST_CORE_INFO("running on device: {0}", m_deviceProps.name);

    m_cudaContext = 0;
//    auto cuRes = cuCtxGetCurrent(&m_cudaContext);
//    if (cuRes != CUDA_SUCCESS)
//        fprintf(stderr, "Error querying current context: error code %d\n", cuRes);

    AST_OPTIX_CHECK(optixDeviceContextCreate(m_cudaContext, 0, &m_optixContext));
    AST_OPTIX_CHECK(optixDeviceContextSetLogCallback
                        (m_optixContext, context_log_cb, nullptr, 4));
}

void Renderer::createModule()
{
    m_moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    m_moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    m_moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    m_pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    m_pipelineCompileOptions.usesMotionBlur = false;
    m_pipelineCompileOptions.numPayloadValues = 2;
    m_pipelineCompileOptions.numAttributeValues = 2;
    m_pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    m_pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";

//    const std::string ptxCode = embedded_ptx_code;

    char log[2048];
    auto sizeof_log = sizeof(log);
    AST_OPTIX_CHECK(optixModuleCreate(m_optixContext,
                                             &m_moduleCompileOptions,
                                             &m_pipelineCompileOptions,
                                             program_ptx_text(),
                                             program_ptx_size,
                                             log,
                                             &sizeof_log,
                                             &m_module));
    if (sizeof_log > 1)
        fprintf(stderr, "Log:\n%s\n", log);
}

void Renderer::createRaygenPG()
{
    m_raygenPGs.resize(1);

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgDesc.raygen.module = m_module;
    pgDesc.raygen.entryFunctionName = "__raygen__renderFrame";

    char log[2048];
    auto sizeof_log = sizeof(log);
    AST_OPTIX_CHECK(optixProgramGroupCreate(m_optixContext,
                                            &pgDesc,
                                            1,
                                            &pgOptions,
                                            log,
                                            &sizeof_log,
                                            &m_raygenPGs[0]));
    if (sizeof_log > 1)
        fprintf(stderr, "Log:\n%s\n", log);
}

void Renderer::createMissPG()
{
    m_missPGs.resize(1);
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pgDesc.miss.module = m_module;
    pgDesc.miss.entryFunctionName = "__miss__radiance";

    char log[2048];
    auto sizeof_log = sizeof(log);
    AST_OPTIX_CHECK(optixProgramGroupCreate(m_optixContext,
                                            &pgDesc,
                                            1,
                                            &pgOptions,
                                            log,
                                            &sizeof_log,
                                            &m_missPGs[0]));
    if (sizeof_log > 1)
        fprintf(stderr, "Log:\n%s\n", log);
}

void Renderer::createHitGroupPG()
{
    m_hitgroupPGs.resize(1);
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pgDesc.hitgroup.moduleCH = m_module;
    pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
    pgDesc.hitgroup.moduleAH = m_module;
    pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";

    char log[2048];
    auto sizeof_log = sizeof(log);
    AST_OPTIX_CHECK(optixProgramGroupCreate(m_optixContext,
                                            &pgDesc,
                                            1,
                                            &pgOptions,
                                            log,
                                            &sizeof_log,
                                            &m_hitgroupPGs[0]));
    if (sizeof_log > 1)
        fprintf(stderr, "Log:\n%s\n", log);
}

void Renderer::createPipeline()
{
    std::vector<OptixProgramGroup> programGroups;
    programGroups.insert(programGroups.end(), m_raygenPGs.begin(), m_raygenPGs.end());
    programGroups.insert(programGroups.end(), m_missPGs.begin(), m_missPGs.end());
    programGroups.insert(programGroups.end(), m_hitgroupPGs.begin(), m_hitgroupPGs.end());

    char log[2048];
    auto sizeof_log = sizeof(log);
    AST_OPTIX_CHECK(optixPipelineCreate(m_optixContext,
                                        &m_pipelineCompileOptions,
                                        &m_pipelineLinkOptions,
                                        programGroups.data(),
                                        programGroups.size(),
                                        log,
                                        &sizeof_log,
                                        &m_pipeline));
    if (sizeof_log > 1)
        fprintf(stderr, "Log:\n%s\n", log);
}

void Renderer::createSBT()
{
    std::vector<RaygenRecord> raygenRecords;
    for (int i = 0; i < m_raygenPGs.size(); i++)
    {
        RaygenRecord rec{};
        AST_OPTIX_CHECK(optixSbtRecordPackHeader(m_raygenPGs[i], &rec));
        rec.data = nullptr; /* for now ... */
        raygenRecords.push_back(rec);
    }
    m_raygenRecords.allocAndUpload(raygenRecords);
    m_sbt.raygenRecord = (CUdeviceptr)m_raygenRecords.m_devicePtr;

    std::vector<MissRecord> missRecords;
    for (int i = 0; i < m_missPGs.size(); i++)
    {
        MissRecord rec{};
        AST_OPTIX_CHECK(optixSbtRecordPackHeader(m_missPGs[i], &rec));
        rec.data = nullptr; /* for now ... */
        missRecords.push_back(rec);
    }
    m_missRecords.allocAndUpload(missRecords);
    m_sbt.missRecordBase = (CUdeviceptr)m_missRecords.devicePtr();
    m_sbt.missRecordStrideInBytes = sizeof(MissRecord);
    m_sbt.missRecordCount = static_cast<int>(missRecords.size());

    std::vector<HitgroupRecord> hitgroupRecords;
    for (int i = 0; i < m_hitgroupPGs.size(); i++)
    {
        HitgroupRecord rec{};
        AST_OPTIX_CHECK(optixSbtRecordPackHeader(m_hitgroupPGs[i], &rec));
        rec.objectID = i; /* for now ... */
        hitgroupRecords.push_back(rec);
    }
    m_hitgroupRecords.allocAndUpload(hitgroupRecords);
    m_sbt.hitgroupRecordBase = (CUdeviceptr)m_hitgroupRecords.devicePtr();
    m_sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    m_sbt.hitgroupRecordCount = static_cast<int>(hitgroupRecords.size());
}