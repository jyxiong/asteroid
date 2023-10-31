#include "asteroid/renderer/scene.h"


using namespace Asteroid;

SceneView::SceneView(const Scene& scene)
    : deviceSpheres(scene.deviceSpheres->data(), scene.deviceSpheres->size()),
      deviceMaterials(scene.deviceMaterials->data(), scene.deviceMaterials->size()){}

void Scene::UpdateDevice()
{
    deviceSpheres = std::make_unique<DeviceBuffer<Sphere>>(spheres);

    deviceMaterials = std::make_unique<DeviceBuffer<Material>>(materials);
}
