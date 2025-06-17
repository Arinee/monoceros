/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     index_logger.cc
 *   \author   Hechong.xyf
 *   \date     May 2018
 *   \version  1.0.0
 *   \brief    Implementation of Mercury Index Logger
 */

#include "index_logger.h"
#include "instance_factory.h"
#include "utility/time_helper.h"
#include <cinttypes>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <thread>

namespace mercury {

/*! Console Logger
 */
struct ConsoleLogger : public IndexLogger
{
    //! Initialize Logger
    virtual int init(const LoggerParams &)
    {
        return 0;
    }

    //! Cleanup Logger
    virtual int cleanup(void)
    {
        return 0;
    }

    //! Log Message
    virtual void log(int level, const char *file, int line, const char *format,
                     va_list args)
    {
        char buffer[8192];
        vsnprintf(buffer, sizeof(buffer), format, args);

        std::ostringstream stream;
        stream << '[' << GetLevelString(level) << ' ' << Realtime::Seconds()
               << ' ' << std::this_thread::get_id() << ' ' << GetBaseName(file)
               << ':' << line << "] " << buffer << '\n';

        if (level <= LEVEL_INFO) {
            std::cout << stream.str() << std::flush;
        } else {
            std::cerr << stream.str() << std::flush;
        }
    }
};

//! Logger Level
int IndexLoggerBroker::_logger_level = 0;

//! Logger
IndexLogger::Pointer IndexLoggerBroker::_logger(new ConsoleLogger);

//! Register Console Logger in Factory
INSTANCE_FACTORY_REGISTER_LOGGER(ConsoleLogger);

} // namespace mercury
