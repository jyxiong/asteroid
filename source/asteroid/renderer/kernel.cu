#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "glm/glm.hpp"

__device__ glm::u8vec4 ConvertToRGBA(const glm::vec4& color)
{
	return static_cast<glm::u8vec4>(color * 255.f);
}


__device__ glm::u8vec4 PerPixel(glm::vec2 coord)
{
    glm::vec3 rayOrigin(0.0f, 0.0f, 1.0f);
	glm::vec3 rayDirection(coord.x, coord.y, -1.0f);

	float radius = 0.5f;

	float a = glm::dot(rayDirection, rayDirection);
	float b = 2.0f * glm::dot(rayOrigin, rayDirection);
	float c = glm::dot(rayOrigin, rayOrigin) - radius * radius;

	float discriminant = b * b - 4.0f * a * c;
	if (discriminant < 0.0f)
		return glm::vec4(0, 0, 0, 1);

	// Quadratic formula:
	// (-b +- sqrt(discriminant)) / 2a

	float closestT = (-b - glm::sqrt(discriminant)) / (2.0f * a);
	float t0 = (-b + glm::sqrt(discriminant)) / (2.0f * a); // Second hit distance (currently unused)

	glm::vec3 hitPoint = rayOrigin + rayDirection * closestT;
	glm::vec3 normal = glm::normalize(hitPoint);

	glm::vec3 lightDir = glm::normalize(glm::vec3(-1, -1, -1));
	float lightIntensity = glm::max(glm::dot(normal, -lightDir), 0.0f); // == cos(angle)

	glm::vec3 sphereColor(1, 0, 1);
	//sphereColor *= lightIntensity;
	return glm::vec4(sphereColor, 1.f);
}

__global__ void cudaProcess(glm::u8vec4* g_odata, int width, int height) {
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
	auto y = blockIdx.y * blockDim.y + threadIdx.y;

	auto uv = glm::vec2(x, y) / glm::vec2(width, height) * 2.f - 1.f;

    g_odata[y * width + x] = ConvertToRGBA(PerPixel(uv));
}

void launch_cudaProcess(dim3 grid, dim3 block, glm::u8vec4* g_odata, int width, int height) {
    cudaProcess<<<grid, block>>>(g_odata, width, height);
}