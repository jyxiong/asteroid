#include "cuda_runtime.h"
#include "glm/glm.hpp"

void launch_cudaProcess(dim3 grid, dim3 block, glm::u8vec4* g_odata, int width, int height);