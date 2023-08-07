#pragma once

#include "asteroid/util/macro.h"
#include "asteroid/event//event.h"

namespace Asteroid
{

class Layer
{
public:
    explicit Layer(std::string name = "Layer");

    virtual ~Layer();

    virtual void OnAttach() {}

    virtual void OnDetach() {}

    virtual void OnUpdate(float ts) {}

    virtual void OnImGuiRender() {}

    inline const std::string &GetName() const { return m_DebugName; }

protected:
    std::string m_DebugName;
};

}