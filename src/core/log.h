#pragma once

#include <memory>

#include "spdlog/spdlog.h"

class Log
{
public:
    static void init();

    inline static std::shared_ptr<spdlog::logger> &getCoreLogger() { return s_core_logger; }

    inline static std::shared_ptr<spdlog::logger> &getClientLogger() { return s_client_logger; }

private:
    static std::shared_ptr<spdlog::logger> s_core_logger;
    static std::shared_ptr<spdlog::logger> s_client_logger;

}; // class Log

// Core log macros
#define LOG_CORE_TRACE(...)    Log::getCoreLogger()->trace(__VA_ARGS__)
#define LOG_CORE_INFO(...)     Log::getCoreLogger()->info(__VA_ARGS__)
#define LOG_CORE_WARN(...)     Log::getCoreLogger()->warn(__VA_ARGS__)
#define LOG_CORE_ERROR(...)    Log::getCoreLogger()->error(__VA_ARGS__)
#define LOG_CORE_CRITICAL(...)    Log::getCoreLogger()->critical(__VA_ARGS__)

// Client log macros
#define LOG_TRACE(...)         Log::getClientLogger()->trace(__VA_ARGS__)
#define LOG_INFO(...)          Log::getClientLogger()->info(__VA_ARGS__)
#define LOG_WARN(...)          Log::getClientLogger()->warn(__VA_ARGS__)
#define LOG_ERROR(...)         Log::getClientLogger()->error(__VA_ARGS__)
#define LOG_CRITICAL(...)         Log::getClientLogger()->critical(__VA_ARGS__)