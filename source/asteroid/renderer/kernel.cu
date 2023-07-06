#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void cudaProcess(uchar4* g_odata, int imgw) {
    extern __shared__ uchar4 sdata[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bw = blockDim.x;
    int bh = blockDim.y;
    int x = blockIdx.x * bw + tx;
    int y = blockIdx.y * bh + ty;

    uchar4 c4 = make_uchar4((x & 0x20) ? 100 : 0, 0, (y & 0x20) ? 100 : 0, 0);
    g_odata[y * imgw + x] = c4;
}

extern "C" void launch_cudaProcess(dim3 grid, dim3 block, int sbytes,
    uchar4* g_odata, int imgw) {
    cudaProcess <<<grid, block, sbytes >> > (g_odata, imgw);
}