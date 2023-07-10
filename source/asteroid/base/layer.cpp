#include "asteroid/base/layer.h"
#include <utility>

using namespace Asteroid;

Layer::Layer(std::string debugName)
    : m_DebugName(std::move(debugName))
{
}

Layer::~Layer() = default;
