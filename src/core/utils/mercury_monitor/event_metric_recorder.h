#ifndef MERCURY_MONITOR_EVENTMETRICRECORDER_H
#define MERCURY_MONITOR_EVENTMETRICRECORDER_H

#include <memory>
#include "bvar/bvar.h"
#include "cat/cat.h"
#include "src/core/framework/index_logger.h"

namespace bvar {
DECLARE_int32(bvar_dump_interval);
} // namespace bvar

namespace mercury {
namespace monitor {

typedef bvar::Window<bvar::IntRecorder, bvar::SERIES_IN_SECOND> RecorderWindow;

static int64_t get_window_recorder_qps(void *arg)
{
    bvar::detail::Sample<bvar::Stat> s;
    static_cast<RecorderWindow *>(arg)->get_span(1, &s);
    // Use floating point to avoid overflow.
    if (s.time_us <= 0) {
        return 0;
    }
    return static_cast<int64_t>(round(s.data.num * 1000000.0 / s.time_us));
}

// Only Support bvar, not for cat
class EventMetricRecorder
{
public:
    EventMetricRecorder()
        : _hit_recorder(),
          _hit_window(&_hit_recorder, bvar::FLAGS_bvar_dump_interval),
          _hit_qps_window(&_hit_recorder, bvar::FLAGS_bvar_dump_interval),
          _hit_qps(get_window_recorder_qps, &_hit_qps_window),
          _failure_window(&_failure_recorder, bvar::FLAGS_bvar_dump_interval),
          _level(0)
    {
    }

    EventMetricRecorder(uint32_t event_level)
        : _hit_recorder(),
          _hit_window(&_hit_recorder, bvar::FLAGS_bvar_dump_interval),
          _hit_qps_window(&_hit_recorder, bvar::FLAGS_bvar_dump_interval),
          _hit_qps(get_window_recorder_qps, &_hit_qps_window),
          _failure_window(&_failure_recorder, bvar::FLAGS_bvar_dump_interval),
          _level(event_level)
    {
    }


    int expose(const std::string &name)
    {
        {
            auto expose_name = "mercury_event_metric_" + name + "_qps";
            int rc = _hit_qps.expose(expose_name);
            if (rc != 0) {
                LOG_ERROR("bvar expose [%s] failed", expose_name.c_str());
                return rc;
            }
            LOG_INFO("bvar expose [%s] successfully", expose_name.c_str());
        }
        {
            auto expose_name = "mercury_event_metric_" + name;
            int rc = _hit_window.expose(expose_name);
            if (0 != rc) {
                LOG_ERROR("bvar expose [%s] failed", expose_name.c_str());
                return rc;
            }
            LOG_INFO("bvar expose [%s] successfully", expose_name.c_str());
        }
        {
            auto expose_name =
                "mercury_event_metric_" + name + "_failure_rate10000";
            int rc = _failure_window.expose(expose_name);
            if (0 != rc) {
                LOG_ERROR("bvar expose [%s] failed", expose_name.c_str());
                return rc;
            }
            LOG_INFO("bvar expose [%s] successfully", expose_name.c_str());
        }

        _cat_name = name;
        // if (FLAGS_mercury_cat_prefix.empty()) {
        //     _cat_type = "mercury";
        // } else {
        //     _cat_type = FLAGS_mercury_cat_prefix + "_mercury";
        // }

        return 0;
    }

    // NB: count = failure_count + succ_count
    void record(int count, int failure_count)
    {
        // for bvar
        _hit_recorder << count;
        if (UNLIKELY(failure_count > 0)) {
            _failure_recorder << (failure_count * RATE_BASE_10000);
        }
        // for cat
        // if (FLAGS_index_partition_is_online &&
        //     FLAGS_mercury_enable_cat_monitor &&
        //     _level >= FLAGS_default_transaction_level) {
        //     if (LIKELY(count == 1 && failure_count == 0)) {
        //         cat::Cat::Instance().LogEvent(_cat_type, _cat_name);
        //     } else {
        //         cat::Cat::Instance().LogBatchEvent(_cat_type, _cat_name, count,
        //                                            failure_count);
        //     }
        // }
    }

    static void dummy(int, int){};

private:
    bvar::IntRecorder _hit_recorder;
    bvar::Window<bvar::IntRecorder> _hit_window;
    RecorderWindow _hit_qps_window;
    bvar::PassiveStatus<int64_t> _hit_qps;

    bvar::IntRecorder _failure_recorder;
    bvar::Window<bvar::IntRecorder> _failure_window;

    std::string _cat_name;
    std::string _cat_type;

    static constexpr int RATE_BASE_10000 = 10000;
    uint32_t _level{ 0 };
    //DF_LOG_DECLARE();
};

//TYPEDEF_SHARED_PTR(EventMetricRecorder);

};
};

#endif // MERCURY_MONITOR_EVENTMETRICRECORDER_H


/* Local Variables: */
/* tab-width: 4 */
/* indent-tabs-mode: nil */
/* coding: utf-8-unix */
/* mode: c++ */
/* End: */
// vim: set tabstop=4 expandtab fileencoding=utf-8 ff=unix ft=cpp norl :