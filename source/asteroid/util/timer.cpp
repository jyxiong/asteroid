#include "asteroid/util/timer.h"

#include <string>
#include <chrono>
#include "asteroid/util/log.h"

using namespace Asteroid;

Timer::Timer()
{
    reset();
}

void Timer::reset()
{
    m_start = std::chrono::high_resolution_clock::now();
}

float Timer::elapsed()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now() - m_start).count() * 0.001f * 0.001f * 0.001f;
}

float Timer::elapsedMillis()
{
    return elapsed() * 1000.0f;
}

ScopedTimer::ScopedTimer(const std::string& name)
    : m_name(name) {}

ScopedTimer::~ScopedTimer()
{
    float time = m_timer.elapsedMillis();
    AST_CORE_INFO("[TIMER] {0} - {1}ms", m_name, time);
}
