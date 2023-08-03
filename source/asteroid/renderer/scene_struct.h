#pragma once

#include <string>
#include <vector>
#include "asteroid/util/vec_math.h"

namespace Asteroid
{

struct RenderState
{
    unsigned int currentIteration{ 0 };
    unsigned int traceDepth{ 1 };
};

struct Camera
{
    float3 position{ 0, 0, 6 };
    float3 direction{ 0, 0, -1 };
    float3 up{ 0, 1, 0 };

    float verticalFov{ 45.f };
    float focalDistance{ 0.f };
    uint2 viewport{ 1, 1 };

    float3 right{};
    float tanHalfFov{};
    float aspectRatio{};

    float2 lastMousePosition{ 0.0f, 0.0f };
};

struct Material
{
    float3 albedo{ 1.0f };
    float roughness = 1.0f;
    float metallic = 0.0f;
    float emittance = 0.0f;

};

struct Sphere
{
    float radius = 0.5f;

    float3 position{ 0.0f };

    int materialIndex = 0;
};

struct Ray
{
    float3 origin;

    float3 direction;
};

struct PathSegment
{
    Ray ray;
    float3 color; // accumulated light
    float3 throughput;
    int pixelIndex;
    unsigned int remainingBounces;
};

struct Intersection
{
    float t;
    float3 normal;
    float3 position;
    int materialIndex;
};

} // namespace Asteroid
