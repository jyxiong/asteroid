#include "asteroid/renderer/scene.h"


using namespace Asteroid;

SceneView::SceneView(const Scene& scene)
    : deviceSpheres(*scene.deviceSpheres),
      deviceMaterials(*scene.deviceMaterials){}

void Scene::UpdateDevice()
{
    deviceSpheres = std::make_shared<DeviceBuffer<Sphere>>(spheres);

    deviceMaterials = std::make_shared<DeviceBuffer<Material>>(materials);
}
