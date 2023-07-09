#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "glm/glm.hpp"

__global__ void cudaProcess(glm::u8vec4* g_odata, int imgw) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bw = blockDim.x;
    int bh = blockDim.y;
    int x = blockIdx.x * bw + tx;
    int y = blockIdx.y * bh + ty;

    g_odata[y * imgw + x] = glm::u8vec4((x & 0x20) ? 100 : 0, 0, (y & 0x20) ? 100 : 0, 255);
}

extern "C" void launch_cudaProcess(dim3 grid, dim3 block, int sbytes,
    glm::u8vec4* g_odata, int imgw) {
    cudaProcess <<<grid, block, sbytes >> > (g_odata, imgw);
}