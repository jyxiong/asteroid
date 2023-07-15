#include <asteroid/util/macro.h>
#include "asteroid/renderer/scene.h"


using namespace Asteroid;

SceneView::SceneView(const Scene& scene)
    : deviceSpheres(scene.deviceSpheres) {}

void Scene::UpdateDevice()
{
    deviceSpheres = DeviceBuffer<Sphere>(spheres.data(), spheres.size(), spheres);

    deviceSpheres.CopyTo(spheres);
}
