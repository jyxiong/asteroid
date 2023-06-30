#pragma once

#include "glm/glm.hpp"

class Texture2D
{
private:
    unsigned int m_id{};
    glm::ivec2 m_resolution{};

public:
    Texture2D();
    ~Texture2D();

    void bind(unsigned int unit = 0) const;
};