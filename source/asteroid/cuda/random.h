#pragma once

#include <cuda_runtime.h>

namespace Asteroid
{
/*! simple 24-bit linear congruence generator */
template<unsigned int N=16>
class LCG
{
public:
    inline __device__ LCG()
    {
    }

    inline __device__ LCG(unsigned int val0, unsigned int val1) { init(val0, val1); }

    inline __device__ void init(unsigned int val0, unsigned int val1)
    {
        unsigned int v0 = val0;
        unsigned int v1 = val1;
        unsigned int s0 = 0;

        for (unsigned int n = 0; n < N; n++)
        {
            s0 += 0x9e3779b9;
            v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
            v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
        }
        state = v0;
    }

    inline __device__ float rand1()
    {
        const unsigned int LCG_A = 1664525u;
        const unsigned int LCG_C = 1013904223u;
        state = (LCG_A * state + LCG_C);
        return (state & 0x00FFFFFF) / (float) 0x01000000;
    }

    inline __device__ glm::vec2 rand2()
    {
        return glm::vec2(rand1(), rand1());
    }

private:
    unsigned int state;
};


}