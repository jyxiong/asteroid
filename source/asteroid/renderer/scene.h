#pragma once

#include <vector>
#include "glm/glm.hpp"
#include "asteroid/cuda/device_buffer.h"
#include "asteroid/renderer/scene_struct.h"

namespace Asteroid
{

struct Scene;

struct SceneView
{

    BufferView<Geometry> deviceGeometries;

    BufferView<Material> deviceMaterials;

    explicit SceneView(const Scene& scene);

};

struct Scene
{

    std::vector<Geometry> geometries;

    std::vector<Material> materials;

    DeviceBuffer<Geometry> deviceGeometries;

    DeviceBuffer<Material> deviceMaterials;

    void UpdateDevice();

};

}
