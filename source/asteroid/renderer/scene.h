#pragma once

#include <vector>
#include "glm/glm.hpp"
#include "asteroid/util/buffer.h"

namespace Asteroid
{

struct Sphere
{
	glm::vec3 Position{0.0f};
	float Radius = 0.5f;

	glm::vec3 Albedo{1.0f};
};

class Scene;
struct SceneView
{
    SceneView(const Scene& scene);

    BufferView<Sphere> spheres;
};

class Scene
{
public:
    void CreateDeviceData();

    SceneView View() const { return SceneView(*this); }

private:

    std::vector<Sphere> m_Spheres;
    Buffer<Sphere> m_DeviceSpheres;

    friend SceneView;
};



}