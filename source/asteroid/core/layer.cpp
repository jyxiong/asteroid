#include <utility>

#include "asteroid/core/layer.h"

using namespace Asteroid;

Layer::Layer(std::string debugName)
    : m_DebugName(std::move(debugName))
{
}

Layer::~Layer() = default;
