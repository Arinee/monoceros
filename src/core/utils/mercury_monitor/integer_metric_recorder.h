#ifndef MERCURY_MONITOR_INTEGERMETRICRECORDER_H
#define MERCURY_MONITOR_INTEGERMETRICRECORDER_H

#include <memory>

#include "bvar/bvar.h"
#include "cat/cat.h"
#include "src/core/framework/index_logger.h"

namespace bvar {
DECLARE_int32(bvar_dump_interval);
DECLARE_bool(bvar_dump);
DECLARE_string(bvar_dump_file);
} // namespace bvar

namespace mercury {
namespace monitor {

// NB: record metric to bvar / cat
// example metric record: failure
class IntegerMetricRecorder
{
public:
    IntegerMetricRecorder()
        : _window(&_int_recorder, bvar::FLAGS_bvar_dump_interval)
    {
    }
    int expose(const std::string &name)
    {
        auto expose_name = "mercury_integer_metric_" + name;
        int rc = _window.expose(expose_name);
        if (0 != rc) {
            LOG_ERROR("bvar expose [%s] failed", expose_name.c_str());
            return rc;
        }
        LOG_INFO("bvar expose [%s] successfully", expose_name.c_str());

        // if (FLAGS_mercury_cat_prefix.empty()) {
        //     _cat_name = "mercury_" + name;
        // } else {
        //     _cat_name = FLAGS_mercury_cat_prefix + "_mercury_" + name;
        // }
        return 0;
    }

    // NB: bvar only support value stored in 43 bit
    void record(long value)
    {
        // for bvar
        _int_recorder << value;
        // // for cat
        // if (FLAGS_index_partition_is_online &&
        //     FLAGS_mercury_enable_cat_monitor) {
        //     cat::Cat::Instance().LogMetricForSum(_cat_name, value);
        // }
    }

    static void dummy(long) {}

private:
    bvar::IntRecorder _int_recorder;
    bvar::Window<bvar::IntRecorder> _window;

    std::string _cat_name;

// private:
//     DF_LOG_DECLARE();
};

// TYPEDEF_SHARED_PTR(IntegerMetricRecorder);

};
};

#endif // MERCURY_MONITOR_INTEGERMETRICRECORDER_H


/* Local Variables: */
/* tab-width: 4 */
/* indent-tabs-mode: nil */
/* coding: utf-8-unix */
/* mode: c++ */
/* End: */
// vim: set tabstop=4 expandtab fileencoding=utf-8 ff=unix ft=cpp norl :