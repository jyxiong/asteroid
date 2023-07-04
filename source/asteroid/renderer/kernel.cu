#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void cudaProcess(unsigned char *g_odata, int imgw) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * imgw);

    g_odata[index * 4] = 1;
    g_odata[index * 4 + 1] = 0;
    g_odata[index * 4 + 2] = 0;
    g_odata[index * 4 + 3] = 1;
}

extern "C" void launch_cudaProcess(dim3 grid, dim3 block, int sbytes,
                                   unsigned char*g_odata, int imgw) {
    cudaProcess<<<grid, block, sbytes>>>(g_odata, imgw);
}