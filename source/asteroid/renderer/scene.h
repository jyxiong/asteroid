#pragma once

namespace Asteroid
{
class SceneView;
class Scene
{
public:
    SceneView View() const;

private:
    friend SceneView;
};

struct SceneView
{
    SceneView(const Scene* scene);
};

}