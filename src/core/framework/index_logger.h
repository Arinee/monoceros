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
#include "alog/Logger.h"
#include "alog/Configurator.h"
#include <cstdarg>
#include <cstring>
#include <memory>

MERCURY_NAMESPACE_BEGIN(core);

/*! Index Logger Broker
 */
class IndexLoggerBroker
{
public:

    //! Log Message
    static alog::Logger *GetLogger()
    {
        static alog::Logger *_logger = alog::Logger::getLogger("mercury");
        return _logger;
    }

private:
    //! Disable them
    IndexLoggerBroker(void) = delete;
    IndexLoggerBroker(const IndexLoggerBroker &) = delete;
    IndexLoggerBroker(IndexLoggerBroker &&) = delete;

    //! Members
    
};

//! Log Debug Message
#ifndef LOG_DEBUG
#define LOG_DEBUG(format, args...)                                              \
    if(__builtin_expect(mercury::core::IndexLoggerBroker::GetLogger()->isLevelEnabled(alog::LOG_LEVEL_DEBUG), 0))      \
        mercury::core::IndexLoggerBroker::GetLogger()->log(alog::LOG_LEVEL_DEBUG, __FILE__, __LINE__, __FUNCTION__, format, ##args);
#endif

//! Log Information Message
#ifndef LOG_INFO
#define LOG_INFO(format, args...)                                               \
    if(__builtin_expect(mercury::core::IndexLoggerBroker::GetLogger()->isLevelEnabled(alog::LOG_LEVEL_INFO), 0))      \
        mercury::core::IndexLoggerBroker::GetLogger()->log(alog::LOG_LEVEL_INFO, __FILE__, __LINE__, __FUNCTION__, format, ##args);
#endif

#ifndef LOG_INFO_INTERVAL
#define LOG_INFO_INTERVAL(log_interval, format, args...)                       \
    do {                                                                       \
        static int log_counter;                                                \
        if (0 == log_counter) {                                                \
            LOG_INFO(format, ##args);                                          \
            log_counter = log_interval;                                        \
        }                                                                      \
        log_counter--;                                                         \
    } while (0)
#endif

//! Log Warn Message
#ifndef LOG_WARN
#define LOG_WARN(format, args...)                                               \
    if(__builtin_expect(mercury::core::IndexLoggerBroker::GetLogger()->isLevelEnabled(alog::LOG_LEVEL_WARN), 0))      \
        mercury::core::IndexLoggerBroker::GetLogger()->log(alog::LOG_LEVEL_WARN, __FILE__, __LINE__, __FUNCTION__, format, ##args);
#endif

//! Log Error Message
#ifndef LOG_ERROR
#define LOG_ERROR(format, args...)                                              \
    if(__builtin_expect(mercury::core::IndexLoggerBroker::GetLogger()->isLevelEnabled(alog::LOG_LEVEL_ERROR), 0))      \
        mercury::core::IndexLoggerBroker::GetLogger()->log(alog::LOG_LEVEL_ERROR, __FILE__, __LINE__, __FUNCTION__, format, ##args);
#endif

//! Log Fatal Message
#ifndef LOG_FATAL
#define LOG_FATAL(format, args...)                                              \
    if(__builtin_expect(mercury::core::IndexLoggerBroker::GetLogger()->isLevelEnabled(alog::LOG_LEVEL_FATAL), 0))      \
        mercury::core::IndexLoggerBroker::GetLogger()->log(alog::LOG_LEVEL_FATAL, __FILE__, __LINE__, __FUNCTION__, format, ##args);
#endif

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

MERCURY_NAMESPACE_END(core);
#endif // __MERCURY_INDEX_LOGGER_H__
