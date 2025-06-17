/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     index_logger.h
 *   \author   Hechong.xyf
 *   \date     May 2018
 *   \version  1.0.0
 *   \brief    Interface of Index Logger
 */

#ifndef __MERCURY_INDEX_LOGGER_H__
#define __MERCURY_INDEX_LOGGER_H__

#include "index_params.h"
#include <cstdarg>
#include <cstring>
#include <memory>

//! Log Debug Message
#ifndef LOG_DEBUG
#define LOG_DEBUG(format, ...)                                                 \
    mercury::IndexLoggerBroker::Log(mercury::IndexLogger::LEVEL_DEBUG,         \
                                    __FILE__, __LINE__, format, ##__VA_ARGS__)
#endif

//! Log Information Message
#ifndef LOG_INFO
#define LOG_INFO(format, ...)                                                  \
    mercury::IndexLoggerBroker::Log(mercury::IndexLogger::LEVEL_INFO,          \
                                    __FILE__, __LINE__, format, ##__VA_ARGS__)
#endif

//! Log Warn Message
#ifndef LOG_WARN
#define LOG_WARN(format, ...)                                                  \
    mercury::IndexLoggerBroker::Log(mercury::IndexLogger::LEVEL_WARN,          \
                                    __FILE__, __LINE__, format, ##__VA_ARGS__)
#endif

//! Log Error Message
#ifndef LOG_ERROR
#define LOG_ERROR(format, ...)                                                 \
    mercury::IndexLoggerBroker::Log(mercury::IndexLogger::LEVEL_ERROR,         \
                                    __FILE__, __LINE__, format, ##__VA_ARGS__)
#endif

//! Log Fatal Message
#ifndef LOG_FATAL
#define LOG_FATAL(format, ...)                                                 \
    mercury::IndexLoggerBroker::Log(mercury::IndexLogger::LEVEL_FATAL,         \
                                    __FILE__, __LINE__, format, ##__VA_ARGS__)
#endif

namespace mercury {

/*! Index Logger Parameters
 */
using LoggerParams = IndexParams;

/*! Index Logger
 */
struct IndexLogger
{
    //! Index Logger Pointer
    typedef std::shared_ptr<IndexLogger> Pointer;

    static const int LEVEL_DEBUG = 0;
    static const int LEVEL_INFO = 1;
    static const int LEVEL_WARN = 2;
    static const int LEVEL_ERROR = 3;
    static const int LEVEL_FATAL = 4;

    //! Retrieve string of level
    static const char *GetLevelString(int level)
    {
        static const char *info[] = { "DEBUG", " INFO", " WARN", "ERROR",
                                      "FATAL" };
        if (level < (int)(sizeof(info) / sizeof(info[0]))) {
            return info[level];
        }
        return "";
    }

    //! Retrieve symbol of level
    static char GetLevelSymbol(int level)
    {
        static const char info[5] = { 'D', 'I', 'W', 'E', 'F' };
        if (level < (int)(sizeof(info) / sizeof(info[0]))) {
            return info[level];
        }
        return ' ';
    }

    //! Retrieve base name of path
    static const char *GetBaseName(const char *filename)
    {
        const char *output = std::strrchr(filename, '/');
        if (!output) {
            output = std::strrchr(filename, '\\');
        }
        return (output ? output + 1 : filename);
    }

    //! Destructor
    virtual ~IndexLogger(void) {}

    //! Initialize Logger
    virtual int init(const LoggerParams &params) = 0;

    //! Cleanup Logger
    virtual int cleanup(void) = 0;

    //! Log Message
    virtual void log(int level, const char *file, int line, const char *format,
                     va_list args) = 0;
};

/*! Index Logger Broker
 */
class IndexLoggerBroker
{
public:
    //! Register Logger
    static void Register(const IndexLogger::Pointer &logger)
    {
        _logger = logger;
    }

    //! Unregister Logger
    static void Unregister(void)
    {
        _logger = nullptr;
    }

    //! Set Level of Logger
    static void SetLevel(int level)
    {
        _logger_level = level;
    }

    //! Log Message
    __attribute__((format(printf, 4, 5))) static void
    Log(int level, const char *file, int line, const char *format, ...)
    {
        if (_logger_level <= level && _logger) {
            va_list args;
            va_start(args, format);
            _logger->log(level, file, line, format, args);
            va_end(args);
        }
    }

private:
    //! Disable them
    IndexLoggerBroker(void) = delete;
    IndexLoggerBroker(const IndexLoggerBroker &) = delete;
    IndexLoggerBroker(IndexLoggerBroker &&) = delete;

    //! Members
    static int _logger_level;
    static IndexLogger::Pointer _logger;
};

} // namespace mercury

#endif // __MERCURY_INDEX_LOGGER_H__
