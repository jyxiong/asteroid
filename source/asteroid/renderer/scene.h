#pragma once

#include <vector>
#include "glm/glm.hpp"
#include "asteroid/util/buffer.h"

namespace Asteroid {

    struct Sphere {
        float Radius = 0.5f;

        glm::vec3 Position{0.0f};

        glm::vec3 Albedo{1.0f};

    };

    struct Scene;

    struct SceneView {

        DeviceBufferView<Sphere> deviceSpheres;

        explicit SceneView(const Scene &scene);

    };

    struct Scene {

        std::vector<Sphere> spheres;

        DeviceBuffer<Sphere> deviceSpheres;

        void UpdateDevice();

    };

}
