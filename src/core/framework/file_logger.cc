/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     file_logger.cc
 *   \author   Hechong.xyf
 *   \date     Mar 2018
 *   \version  1.0.0
 *   \brief    Implementation of Mercury File Logger
 */

#include "index_error.h"
#include "instance_factory.h"
#include "utility/time_helper.h"
#include <sstream>
#include <thread>

MERCURY_NAMESPACE_BEGIN(core);

static const char kParamLoggerPath[] = "file.logger.path";

/*! File Logger
 */
struct FileLogger : public IndexLogger
{
    //! Constructor
    FileLogger(void) : _file(stdout) {}

    //! Destructor
    ~FileLogger(void)
    {
        if (_file && _file != stdout && _file != stderr) {
            fclose(_file);
        }
    }

    //! Initialize Logger
    virtual int init(const LoggerParams &params)
    {
        std::string path;
        params.get(kParamLoggerPath, &path);

        if (!path.empty()) {
            _file = fopen(path.c_str(), "w");
            if (_file == nullptr) {
                return IndexError_IO;
            }
        }
        return 0;
    }

    //! Cleanup Logger
    virtual int cleanup(void)
    {
        if (_file && _file != stdout && _file != stderr) {
            fclose(_file);
        }
        _file = nullptr;
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

        fwrite(stream.str().c_str(), stream.str().size(), 1, _file);
        fflush(_file);
    }

    //! Members
    FILE *_file;
};

//! Register File Logger in Factory
INSTANCE_FACTORY_REGISTER_LOGGER(FileLogger);

MERCURY_NAMESPACE_END(core);
