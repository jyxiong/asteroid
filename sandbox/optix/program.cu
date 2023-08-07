// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include <optix_device.h>
#include "launchParams.h"

using namespace Asteroid;

/*! launch parameters in constant memory, filled in by optix upon
    optixLaunch (this gets filled in from the buffer we pass to
    optixLaunch) */
extern "C" __constant__ LaunchParams optixLaunchParams;

// for this simple example, we have a single ray type
enum
{
    SURFACE_RAY_TYPE = 0, RAY_TYPE_COUNT
};

static __forceinline__ __device__
void *unpackPointer(uint32_t i0, uint32_t i1)
{
    const uint64_t uptr = static_cast<uint64_t>( i0 ) << 32 | i1;
    void *ptr = reinterpret_cast<void *>( uptr );
    return ptr;
}

static __forceinline__ __device__
void packPointer(void *ptr, uint32_t &i0, uint32_t &i1)
{
    const uint64_t uptr = reinterpret_cast<uint64_t>( ptr );
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

template<typename T>
static __forceinline__ __device__ T *getPRD()
{
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return reinterpret_cast<T *>( unpackPointer(u0, u1));
}

//------------------------------------------------------------------------------
// closest hit and anyhit programs for radiance-type rays.
//
// Note eventually we will have to create one pair of those for each
// ray type and each geometry type we want to render; but this
// simple example doesn't use any actual geometries yet, so we only
// create a single, dummy, set of them (we do have to have at least
// one group of them to set up the SBT)
//------------------------------------------------------------------------------

extern "C" __global__ void __closesthit__radiance()
{
    const TriangleMeshSBTData &sbtData
        = *(const TriangleMeshSBTData *) optixGetSbtDataPointer();

    // compute normal:
    auto primID = optixGetPrimitiveIndex();
    const glm::ivec3 index = sbtData.index[primID];
    const glm::vec3 &A = sbtData.vertex[index.x];
    const glm::vec3 &B = sbtData.vertex[index.y];
    const glm::vec3 &C = sbtData.vertex[index.z];
    const glm::vec3 Ng = glm::normalize(glm::cross(B - A, C - A));

    auto dir = optixGetWorldRayDirection();
    const glm::vec3 rayDir = { dir.x, dir.y, dir.z };
    const float cosDN = 0.2f + .8f * fabsf(glm::dot(rayDir, Ng));
    glm::vec3 &prd = *(glm::vec3 *) getPRD<glm::vec3>();
    prd = cosDN * sbtData.color;
}

extern "C" __global__ void __anyhit__radiance() { /*! for this simple example, this will remain empty */ }



//------------------------------------------------------------------------------
// miss program that gets called for any ray that did not have a
// valid intersection
//
// as with the anyhit/closest hit programs, in this example we only
// need to have _some_ dummy function to set up a valid SBT
// ------------------------------------------------------------------------------

extern "C" __global__ void __miss__radiance()
{
    auto &prd = *(glm::vec3 *) getPRD<glm::vec3>();
    // set to constant white as background color
    prd = glm::vec3(1.f);
}

//------------------------------------------------------------------------------
// ray gen program - the actual rendering happens in here
//------------------------------------------------------------------------------
extern "C" __global__ void __raygen__renderFrame()
{
// compute a test pattern based on pixel ID
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;

    const auto &camera = optixLaunchParams.camera;

    // our per-ray data for this example. what we initialize it to
    // won't matter, since this value will be overwritten by either
    // the miss or hit program, anyway
    auto pixelColorPRD = glm::vec3(0.f);

    // the values we store the PRD pointer in:
    uint32_t u0, u1;
    packPointer(&pixelColorPRD, u0, u1);

    // normalized screen plane position, in [0,1]^2
    auto uv = glm::vec2(ix, iy) / glm::vec2(optixLaunchParams.frame.size) * 2.f - 1.f;

    // generate ray direction
    auto rayDir = glm::normalize(camera.direction
                                     + uv.x * camera.horizontal
                                     + uv.y * camera.vertical);

    optixTrace(optixLaunchParams.traversable,
               { camera.position.x, camera.position.y, camera.position.z },
               { rayDir.x, rayDir.y, rayDir.z },
               0.f,    // tmin
               1e20f,  // tmax
               0.0f,   // rayTime
               OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_DISABLE_ANYHIT,//OPTIX_RAY_FLAG_NONE,
               SURFACE_RAY_TYPE,             // SBT offset
               RAY_TYPE_COUNT,               // SBT stride
               SURFACE_RAY_TYPE,             // missSBTIndex 
               u0, u1);

    // and write to frame buffer ...
    const uint32_t fbIndex = ix + iy * optixLaunchParams.frame.size.x;
    optixLaunchParams.frame.colorBuffer[fbIndex] = glm::u8vec4(pixelColorPRD * 255.f, 255);
}
