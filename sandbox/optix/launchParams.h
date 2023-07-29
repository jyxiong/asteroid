#pragma once

namespace Asteroid
{
struct LaunchParams
{
    int frameID{ 0 };
    unsigned int *colorBuffer;
    int width, height;
};
}