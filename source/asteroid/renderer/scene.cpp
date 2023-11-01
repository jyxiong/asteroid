#include "asteroid/renderer/scene.h"

using namespace Asteroid;

SceneView::SceneView(const Scene &scene)
    : deviceGeometries(scene.deviceGeometries.data(), scene.deviceGeometries.size()),
      deviceMaterials(scene.deviceMaterials.data(), scene.deviceMaterials.size())
{
}

void Scene::UpdateDevice()
{
    for (auto& geometry : geometries)
    {
        geometry.updateTransform();
    }

    deviceGeometries.upload(geometries);

    deviceMaterials.upload(materials);
}
