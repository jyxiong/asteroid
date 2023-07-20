#pragma once

#include <vector>
#include "glm/glm.hpp"
#include "asteroid/util/buffer.h"

namespace Asteroid {

    struct Material {
        glm::vec3 Albedo{ 1.0f };
        float Roughness = 1.0f;
        float Metallic = 0.0f;
    };

    struct Sphere {
        float Radius = 0.5f;

        glm::vec3 Position{0.0f};

        int MaterialId = 0;

    };

    struct Scene;

    struct SceneView {

        BufferView<Sphere> deviceSpheres;

        BufferView<Material> deviceMaterials;

        explicit SceneView(const Scene &scene);

    };

    struct Scene {

        std::vector<Sphere> spheres;

        std::vector<Material> materials;

        std::unique_ptr<Buffer<Sphere>> deviceSpheres;

        std::unique_ptr<Buffer<Material>> deviceMaterials;

        void UpdateDevice();

    };

}
