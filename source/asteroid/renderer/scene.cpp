#include "asteroid/renderer/scene.h"

using namespace Asteroid;

SceneView::SceneView(const Scene& scene)
    : spheres(scene.deviceSpheres) {}

void Scene::CreateDeviceData()
{
    deviceSpheres = Buffer<Sphere>(spheres.data(), spheres.size());
}
