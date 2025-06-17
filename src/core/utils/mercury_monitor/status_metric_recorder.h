#ifndef MERCURY_MONITOR_STATUSMETRICRECORDER_H
#define MERCURY_MONITOR_STATUSMETRICRECORDER_H

#include <memory>
#include "bvar/bvar.h"
#include "cat/cat.h"
#include "src/core/framework/index_logger.h"

namespace mercury {
namespace monitor {

static long get_value_from_status(void *arg)
{
    return static_cast<bvar::Status<long> *>(arg)->get_value();
}

// NB: Support for status value, use passive status to plot
// example: segment_count
class StatusMetricRecorder
{
public:
    StatusMetricRecorder()
        : _status(), _passive_status(get_value_from_status, &_status)
    {
    }

    int expose(const std::string &name)
    {
        auto expose_name = "mercury_status_metric_" + name;
        int rc = _passive_status.expose(expose_name);
        if (0 != rc) {
            LOG_ERROR("bvar expose [%s] failed", expose_name.c_str());
            return rc;
        }
        LOG_ERROR("bvar expose [%s] successfully", expose_name.c_str());

        // if (FLAGS_mercury_cat_prefix.empty()) {
        //     _cat_name = "mercury_" + name;
        // } else {
        //     _cat_name = FLAGS_mercury_cat_prefix + "_mercury_" + name;
        // }
        return 0;
    }

    void record(long status)
    {
        // for bvar
        _status.set_value(status);
        // for cat
        // if (FLAGS_index_partition_is_online &&
        //     FLAGS_mercury_enable_cat_monitor) {
        //     cat::Cat::Instance().LogMetricForSum(_cat_name, status);
        // }
    }

    static void dummy(long) {}

    long get_value(void)
    {
        return _status.get_value();
    }

private:
    bvar::Status<long> _status;
    bvar::PassiveStatus<long> _passive_status;

    std::string _name;
    std::string _cat_name;

// private:
//     DF_LOG_DECLARE();
};

//TYPEDEF_SHARED_PTR(StatusMetricRecorder);

};
};

#endif // MERCURY_MONITOR_STATUSMETRICRECORDER_H


/* Local Variables: */
/* tab-width: 4 */
/* indent-tabs-mode: nil */
/* coding: utf-8-unix */
/* mode: c++ */
/* End: */
// vim: set tabstop=4 expandtab fileencoding=utf-8 ff=unix ft=cpp norl :