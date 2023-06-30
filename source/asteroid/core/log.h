#pragma once

#include <memory>
#include "spdlog/spdlog.h"

namespace Asteroid
{

class Log
{
public:
    static void Init();

    inline static std::shared_ptr<spdlog::logger> &GetCoreLogger() { return s_CoreLogger; }

    inline static std::shared_ptr<spdlog::logger> &GetClientLogger() { return s_ClientLogger; }

private:
    static std::shared_ptr<spdlog::logger> s_CoreLogger;
    static std::shared_ptr<spdlog::logger> s_ClientLogger;
};

}

// Core log macros
#define AST_CORE_TRACE(...)    ::Asteroid::Log::GetCoreLogger()->trace(__VA_ARGS__)
#define AST_CORE_INFO(...)     ::Asteroid::Log::GetCoreLogger()->info(__VA_ARGS__)
#define AST_CORE_WARN(...)     ::Asteroid::Log::GetCoreLogger()->warn(__VA_ARGS__)
#define AST_CORE_ERROR(...)    ::Asteroid::Log::GetCoreLogger()->error(__VA_ARGS__)
#define AST_CORE_FATAL(...)    ::Asteroid::Log::GetCoreLogger()->fatal(__VA_ARGS__)

// Client log macros
#define AST_TRACE(...)          ::Asteroid::Log::GetClientLogger()->trace(__VA_ARGS__)
#define AST_INFO(...)          ::Asteroid::Log::GetClientLogger()->info(__VA_ARGS__)
#define AST_WARN(...)          ::Asteroid::Log::GetClientLogger()->warn(__VA_ARGS__)
#define AST_ERROR(...)          ::Asteroid::Log::GetClientLogger()->error(__VA_ARGS__)
#define AST_FATAL(...)          ::Asteroid::Log::GetClientLogger()->fatal(__VA_ARGS__)