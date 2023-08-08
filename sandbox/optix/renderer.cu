#include "renderer.h"

// this include may only appear in a single source file:
#include <optix_function_table_definition.h>
#include "glm/gtx/transform.hpp"
#include "asteroid/util/log.h"
#include "otkHelloKernelPTX.h"

using namespace Asteroid;

//extern "C" char embedded_ptx_code[];

void TriangleMesh::addCube(const glm::vec3 &center, const glm::vec3 &size)
{
    auto transform = glm::translate(center) * glm::scale(size);
    addUnitCube(transform);
}

/*! add a unit cube (subject to given xfm matrix) to the current
    triangleMesh */
void TriangleMesh::addUnitCube(const glm::mat4 &transform)
{
    int firstVertexID = (int) vertex.size();
    vertex.emplace_back(transform * glm::vec4(0.f, 0.f, 0.f, 1.f));
    vertex.emplace_back(transform * glm::vec4(1.f, 0.f, 0.f, 1.f));
    vertex.emplace_back(transform * glm::vec4(0.f, 1.f, 0.f, 1.f));
    vertex.emplace_back(transform * glm::vec4(1.f, 1.f, 0.f, 1.f));
    vertex.emplace_back(transform * glm::vec4(0.f, 0.f, 1.f, 1.f));
    vertex.emplace_back(transform * glm::vec4(1.f, 0.f, 1.f, 1.f));
    vertex.emplace_back(transform * glm::vec4(0.f, 1.f, 1.f, 1.f));
    vertex.emplace_back(transform * glm::vec4(1.f, 1.f, 1.f, 1.f));

    int indices[] = { 0, 1, 3, 2, 3, 0,
                      5, 7, 6, 5, 6, 4,
                      0, 4, 5, 0, 5, 1,
                      2, 3, 7, 2, 7, 6,
                      1, 5, 7, 1, 7, 3,
                      4, 0, 2, 4, 2, 6 };
    for (int i = 0; i < 12; i++)
        index.push_back(firstVertexID + glm::ivec3(indices[3 * i + 0],
                                                   indices[3 * i + 1],
                                                   indices[3 * i + 2]));
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
    TriangleMeshSBTData data;
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

    m_colorBuffer.resize(width * height * sizeof(glm::u8vec4));

    m_launchParams.frame.size = glm::ivec2(width, height);
    m_launchParams.frame.colorBuffer = (glm::u8vec4 *) m_colorBuffer.devicePtr();
}

void Renderer::Render()
{
    if (m_launchParams.frame.size.x == 0) return;

    m_launchParamsBuffer.upload(&m_launchParams, 1);

    AST_OPTIX_CHECK(optixLaunch(/*! pipeline we're launching launch: */
        m_pipeline, m_stream,
        /*! parameters and SBT */
        (CUdeviceptr) m_launchParamsBuffer.devicePtr(),
        m_launchParamsBuffer.m_sizeInBytes,
        &m_sbt,
        /*! dimensions of the launch: */
        m_launchParams.frame.size.x,
        m_launchParams.frame.size.y,
        1
    ))
    // sync - make sure the frame is rendered before we download and
    // display (obviously, for a high-performance application you
    // want to use streams and double-buffering, but for this simple
    // example, this will have to do)
    AST_CUDA_SYNC_CHECK()

    m_finalImage->SetData(m_colorBuffer.devicePtr());
}

void Renderer::setModel(const std::vector<TriangleMesh> &model)
{
    m_meshes = model;

    createAccel();

    AST_CORE_INFO("creating optix shader binding table ...");
    createSBT();
}

void Renderer::setCamera(const Camera &camera)
{
    m_camera = camera;
    m_launchParams.camera.position = m_camera.position;
    m_launchParams.camera.direction = m_camera.direction;
    const float cosFovy = 0.66f;
    const float aspect = m_camera.aspectRatio;
    m_launchParams.camera.horizontal = cosFovy * aspect * m_camera.right;
    m_launchParams.camera.vertical = cosFovy * m_camera.up;
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

    AST_OPTIX_CHECK(optixInit())

    AST_CORE_INFO("initialize optix successfully!");
}

void Renderer::createContext()
{
    const int deviceID = 0;
    AST_CUDA_CHECK(cudaSetDevice(deviceID))
    AST_CUDA_CHECK(cudaStreamCreate(&m_stream))

    cudaGetDeviceProperties(&m_deviceProps, deviceID);
    AST_CORE_INFO("running on device: {0}", m_deviceProps.name);

    m_cudaContext = nullptr;
//    auto cuRes = cuCtxGetCurrent(&m_cudaContext);
//    if (cuRes != CUDA_SUCCESS)
//        fprintf(stderr, "Error querying current context: error code %d\n", cuRes);

    m_optixContext.create();

//    AST_OPTIX_CHECK(optixDeviceContextCreate(m_cudaContext, nullptr, &m_optixContext))
//    AST_OPTIX_CHECK(optixDeviceContextSetLogCallback
//                        (m_optixContext, context_log_cb, nullptr, 4))
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

    m_pipelineLinkOptions.maxTraceDepth          = 2;

//    const std::string ptxCode = embedded_ptx_code;

    char log[2048];
    auto sizeof_log = sizeof(log);
    AST_OPTIX_CHECK(optixModuleCreate((OptixDeviceContext)m_optixContext,
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
    AST_OPTIX_CHECK(optixProgramGroupCreate((OptixDeviceContext)m_optixContext,
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
    AST_OPTIX_CHECK(optixProgramGroupCreate((OptixDeviceContext)m_optixContext,
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
    AST_OPTIX_CHECK(optixProgramGroupCreate((OptixDeviceContext)m_optixContext,
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
    AST_OPTIX_CHECK(optixPipelineCreate((OptixDeviceContext)m_optixContext,
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
    m_sbt.raygenRecord = (CUdeviceptr) m_raygenRecords.m_devicePtr;

    std::vector<MissRecord> missRecords;
    for (int i = 0; i < m_missPGs.size(); i++)
    {
        MissRecord rec{};
        AST_OPTIX_CHECK(optixSbtRecordPackHeader(m_missPGs[i], &rec));
        rec.data = nullptr; /* for now ... */
        missRecords.push_back(rec);
    }
    m_missRecords.allocAndUpload(missRecords);
    m_sbt.missRecordBase = (CUdeviceptr) m_missRecords.devicePtr();
    m_sbt.missRecordStrideInBytes = sizeof(MissRecord);
    m_sbt.missRecordCount = static_cast<int>(missRecords.size());

    int numObjects = m_meshes.size();
    std::vector<HitgroupRecord> hitgroupRecords;
    for (int i = 0; i < numObjects; i++)
    {
        // we only have a single object type so far
        int objectType = 0;
        HitgroupRecord rec;
        AST_OPTIX_CHECK(optixSbtRecordPackHeader(m_hitgroupPGs[objectType], &rec));
        rec.data.vertex = (glm::vec3 *) m_vertexBuffer[i].devicePtr();
        rec.data.index = (glm::ivec3 *) m_indexBuffer[i].devicePtr();
        rec.data.color = m_meshes[i].color;
        hitgroupRecords.push_back(rec);
    }
    m_hitgroupRecords.allocAndUpload(hitgroupRecords);
    m_sbt.hitgroupRecordBase = (CUdeviceptr) m_hitgroupRecords.devicePtr();
    m_sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    m_sbt.hitgroupRecordCount = (int) hitgroupRecords.size();
}

void Renderer::createAccel()
{

    m_vertexBuffer.resize(m_meshes.size());
    m_indexBuffer.resize(m_meshes.size());

    std::vector<OptixBuildInput> triangleInput(m_meshes.size());
    std::vector<CUdeviceptr> d_vertices(m_meshes.size());
    std::vector<CUdeviceptr> d_indices(m_meshes.size());
    std::vector<uint32_t> triangleInputFlags(m_meshes.size());

    for (size_t meshId = 0; meshId < m_meshes.size(); ++meshId)
    {
        TriangleMesh &model = m_meshes[meshId];

        // upload the model to the device: the builder
        m_vertexBuffer[meshId].allocAndUpload(model.vertex);
        m_indexBuffer[meshId].allocAndUpload(model.index);

        // ==================================================================
        // triangle inputs
        // ==================================================================
        triangleInput[meshId] = {};
        triangleInput[meshId].type
            = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

        // create local variables, because we need a *pointer* to the
        // device pointers
        d_vertices[meshId] = (CUdeviceptr) m_vertexBuffer[meshId].devicePtr();
        d_indices[meshId] = (CUdeviceptr) m_indexBuffer[meshId].devicePtr();

        triangleInput[meshId].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        triangleInput[meshId].triangleArray.vertexStrideInBytes = sizeof(glm::vec3);
        triangleInput[meshId].triangleArray.numVertices = (int) model.vertex.size();
        triangleInput[meshId].triangleArray.vertexBuffers = &d_vertices[meshId];

        triangleInput[meshId].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        triangleInput[meshId].triangleArray.indexStrideInBytes = sizeof(glm::ivec3);
        triangleInput[meshId].triangleArray.numIndexTriplets = (int) model.index.size();
        triangleInput[meshId].triangleArray.indexBuffer = d_indices[meshId];

        triangleInputFlags[meshId] = { 0 };

        // in this example we have one SBT entry, and no per-primitive
        // materials:
        triangleInput[meshId].triangleArray.flags = &triangleInputFlags[meshId];
        triangleInput[meshId].triangleArray.numSbtRecords = 1;
        triangleInput[meshId].triangleArray.sbtIndexOffsetBuffer = 0;
        triangleInput[meshId].triangleArray.sbtIndexOffsetSizeInBytes = 0;
        triangleInput[meshId].triangleArray.sbtIndexOffsetStrideInBytes = 0;

    }

    // ==================================================================
    // BLAS setup
    // ==================================================================

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE
        | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accelOptions.motionOptions.numKeys = 1;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes blasBufferSizes;
    AST_OPTIX_CHECK(optixAccelComputeMemoryUsage
                        ((OptixDeviceContext)m_optixContext,
                         &accelOptions,
                         triangleInput.data(),
                         triangleInput.size(),  // num_build_inputs
                         &blasBufferSizes
                        ));

    // ==================================================================
    // prepare compaction
    // ==================================================================

    DeviceBuffer compactedSizeBuffer;
    compactedSizeBuffer.alloc(sizeof(uint64_t));

    OptixAccelEmitDesc emitDesc;
    emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitDesc.result = (CUdeviceptr) compactedSizeBuffer.devicePtr();

    // ==================================================================
    // execute build (main stage)
    // ==================================================================

    DeviceBuffer tempBuffer;
    tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);

    DeviceBuffer outputBuffer;
    outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);

    AST_OPTIX_CHECK(optixAccelBuild((OptixDeviceContext)m_optixContext,
        /* stream */0,
                                    &accelOptions,
                                    triangleInput.data(),
                                    triangleInput.size(),
                                    (CUdeviceptr) tempBuffer.devicePtr(),
                                    tempBuffer.m_sizeInBytes,

                                    (CUdeviceptr) outputBuffer.devicePtr(),
                                    outputBuffer.m_sizeInBytes,

                                    &m_launchParams.traversable,

                                    &emitDesc, 1
    ));
    AST_CUDA_SYNC_CHECK();

    // ==================================================================
    // perform compaction
    // ==================================================================
    uint64_t compactedSize;
    compactedSizeBuffer.download(&compactedSize, 1);

    m_asBuffer.alloc(compactedSize);
    AST_OPTIX_CHECK(optixAccelCompact((OptixDeviceContext)m_optixContext,
        /*stream:*/0,
                                      m_launchParams.traversable,
                                      (CUdeviceptr) m_asBuffer.devicePtr(),
                                      m_asBuffer.m_sizeInBytes,
                                      &m_launchParams.traversable));
    AST_CUDA_SYNC_CHECK();

    // ==================================================================
    // aaaaaand .... clean up
    // ==================================================================
    outputBuffer.free(); // << the UNcompacted, temporary output buffer
    tempBuffer.free();
    compactedSizeBuffer.free();
}
