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
    BufferView<AreaLight> deviceAreaLights;

    BufferView<Geometry> deviceGeometries;

    BufferView<Material> deviceMaterials;

    explicit SceneView(const Scene& scene);

};

struct Scene
{
    std::vector<AreaLight> areaLights;

    std::vector<Geometry> geometries;

    std::vector<Material> materials;

    DeviceBuffer<AreaLight> deviceAreaLights;

    DeviceBuffer<Geometry> deviceGeometries;

    DeviceBuffer<Material> deviceMaterials;

    void updateDevice();

};

}
