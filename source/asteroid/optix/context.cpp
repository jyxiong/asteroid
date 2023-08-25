#include "asteroid/optix/context.h"

#include <iomanip>
#include <iostream>
#include <cuda_runtime.h>
#include "asteroid/util/log.h"
#include "asteroid/util/macro.h"

using namespace Asteroid;

// --------------------------------------------------------------------
static void context_log_cb(unsigned int level,
                           const char *tag,
                           const char *message,
                           void *)
{
    AST_CORE_INFO("{0}{1}:{2}", (int) level, tag, message);
}

// --------------------------------------------------------------------
Context::Context()
    : m_device_id(0)
{
    m_options =
        {
            &context_log_cb,                     // logCallbackFunction
            nullptr,                                 // logCallbackData
            4,                                       // logCallbackLevel
            OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL // validationMode
        };
}

Context::Context(const OptixDeviceContextOptions &options)
    : m_options(options)
{

}

Context::Context(unsigned int device_id)
    : m_device_id(device_id)
{
    m_options =
        {
            &context_log_cb,                     // logCallbackFunction
            nullptr,                                 // logCallbackData
            4,                                       // logCallbackLevel
            OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL // validationMode
        };
}

Context::Context(unsigned int device_id, const OptixDeviceContextOptions &options)
    : m_device_id(device_id), m_options(options)
{

}

// --------------------------------------------------------------------
void Context::create()
{
    /// Verify if the \c device_id exceeds the detected number of GPU devices.
    int32_t num_device;
    AST_CUDA_CHECK(cudaGetDeviceCount(&num_device));
    AST_ASSERT((int32_t) m_device_id < num_device, "The device ID exceeds the detected number of GPU devices.");

    // Set device with specified id.
    cudaDeviceProp prop{};
    AST_CUDA_CHECK(cudaGetDeviceProperties(&prop, m_device_id));
    AST_CUDA_CHECK(cudaSetDevice(m_device_id));

    CUcontext cu_ctx = nullptr;
    AST_OPTIX_CHECK(optixDeviceContextCreate(cu_ctx, &m_options, &m_ctx));
}

void Context::destroy()
{
    if (m_ctx) AST_OPTIX_CHECK(optixDeviceContextDestroy(m_ctx));
    m_ctx = nullptr;
}

// --------------------------------------------------------------------
void Context::setOptions(const OptixDeviceContextOptions &options)
{
    m_options = options;
}
void Context::setLogCallbackFunction(OptixLogCallback callback_func)
{
    m_options.logCallbackFunction = callback_func;
}
void Context::setLogCallbackData(void *callback_data)
{
    m_options.logCallbackData = callback_data;
}
void Context::setLogCallbackLevel(int callback_level)
{
    m_options.logCallbackLevel = callback_level;
}
void Context::enableValidation()
{
    m_options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
}

void Context::disableValidation()
{
    m_options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF;
}

OptixDeviceContextOptions Context::options() const
{
    return m_options;
}

// --------------------------------------------------------------------
void Context::setDeviceId(const unsigned int device_id)
{
    m_device_id = device_id;
}
unsigned int Context::deviceId() const
{
    return m_device_id;
}
