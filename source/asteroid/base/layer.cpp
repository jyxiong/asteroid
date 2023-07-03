#include <utility>

#include "asteroid/base/layer.h"

using namespace Asteroid;

Layer::Layer(std::string debugName)
    : m_DebugName(std::move(debugName))
{
}

Layer::~Layer() = default;
