#include "asteroid/renderer/scene.h"

using namespace Asteroid;

SceneView::SceneView(const Scene& scene)
    : spheres(scene.m_DeviceSpheres){}

void Scene::CreateDeviceData()
{
    m_DeviceSpheres = Buffer<Sphere>(m_Spheres.data(), m_Spheres.size());
}
