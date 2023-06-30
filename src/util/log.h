#pragma once

#include <memory>

#include "spdlog/spdlog.h"

class Log
{
public:
    Log() = default;
    ~Log() = default;

    static void init();

    inline static std::shared_ptr<spdlog::logger> getCoreLogger() { return s_coreLogger; }
    inline static std::shared_ptr<spdlog::logger> getClientLogger() { return s_clientLogger; }

private:
    static std::shared_ptr<spdlog::logger> s_coreLogger;
    static std::shared_ptr<spdlog::logger> s_clientLogger;

}; // class Log
