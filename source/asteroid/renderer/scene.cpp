#include "asteroid/renderer/scene.h"


using namespace Asteroid;

SceneView::SceneView(const Scene& scene)
    : deviceSpheres(*scene.deviceSpheres) {}

void Scene::UpdateDevice()
{
    deviceSpheres = std::make_shared<DeviceBuffer<Sphere>>(spheres);
}
