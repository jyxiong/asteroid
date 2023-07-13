#include "asteroid/renderer/scene.h"

using namespace Asteroid;

SceneView Scene::View() const
{
    return SceneView(this);
}

SceneView::SceneView(const Scene *scene) {}
