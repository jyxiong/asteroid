#pragma once

#include "asteroid/util/vec_math.h"

namespace Asteroid
{
struct LaunchParams
{
    struct
    {
        unsigned int *colorBuffer;
        int2 size;
    } frame;

    struct
    {
        float3 position;
        float3 direction;
        float3 horizontal;
        float3 vertical;
    } camera;

    OptixTraversableHandle traversable;
};
}