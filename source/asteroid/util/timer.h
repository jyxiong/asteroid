#pragma once

#include <iostream>
#include <string>
#include <chrono>
#include "asteroid/util/log.h"

namespace Asteroid
{

class Timer
{
public:
    Timer();

    void reset();

    float elapsed();

    float elapsedMillis();

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> m_start;

}; // class Timer

class ScopedTimer
{
public:
    explicit ScopedTimer(const std::string& name);
    ~ScopedTimer();

private:
    std::string m_name;
    Timer m_timer;

}; // class ScopedTimer

} // namespace Asteroid