#pragma once

#include <chrono>

class StopWatch
{
public:
    StopWatch() = default;
    ~StopWatch() = default;

    void start() { m_beginTime = std::chrono::system_clock::now(); }
    void stop() { m_endTime = std::chrono::system_clock::now(); }

    std::int64_t getNanoseconds()
    {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(m_endTime - m_beginTime).count();
    }

    std::int64_t getMicroseconds()
    {
        return std::chrono::duration_cast<std::chrono::microseconds>(m_endTime - m_beginTime).count();
    }

    std::int64_t getMilliseconds()
    {
        return std::chrono::duration_cast<std::chrono::milliseconds>(m_endTime - m_beginTime).count();
    }

    std::int64_t getSeconds()
    {
        return std::chrono::duration_cast<std::chrono::seconds>(m_endTime - m_beginTime).count();
    }

private:
    std::chrono::system_clock::time_point m_beginTime;
    std::chrono::system_clock::time_point m_endTime;

}; // class StopWatch
