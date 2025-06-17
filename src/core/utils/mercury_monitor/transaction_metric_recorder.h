#ifndef MERCURY_MONITOR_TRANSACTIONMETRICRECORDER_H
#define MERCURY_MONITOR_TRANSACTIONMETRICRECORDER_H

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

class TransactionMetricRecorder
{
public:
    TransactionMetricRecorder()
        : _failure_window(&_failure_recorder, bvar::FLAGS_bvar_dump_interval)
    {
        _level = 0;
    }

    TransactionMetricRecorder(uint32_t transaction_level)
        : _failure_window(&_failure_recorder, bvar::FLAGS_bvar_dump_interval),
          _level(transaction_level)
    {
    }

    int expose(const std::string &name)
    {
        {
            auto expose_name = "mercury_transaction_metric_" + name;
            auto rc = _latency_recorder.expose(expose_name);
            if (0 != rc) {
                LOG_ERROR("bvar expose [%s] failed", expose_name.c_str());
                return rc;
            }
            LOG_INFO("bvar expose [%s] successfully", expose_name.c_str());
        }
        {
            auto expose_name =
                "mercury_transaction_metric_" + name + "_failure_rate10000";
            auto rc = _failure_window.expose(expose_name);
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

    void record(int latency_us, bool succ)
    {
        _latency_recorder << latency_us;
        _failure_recorder << (succ ? 0 : RATE_BASE_10000);

        // if (FLAGS_index_partition_is_online &&
        //     FLAGS_mercury_enable_cat_monitor &&
        //     _level >= FLAGS_default_transaction_level) {
        //     auto transaction =
        //         cat::Cat::Instance().NewTransaction(_cat_type, _cat_name);
        //     transaction->SetStatus(succ ? "0" : "FAILED");
        //     transaction->SetDuration(latency_us); // time in us
        //     transaction->Complete();
        // }
    }

    static void dummy(int, bool) {}

private:
    bvar::LatencyRecorder _latency_recorder;
    bvar::IntRecorder _failure_recorder;
    bvar::Window<bvar::IntRecorder> _failure_window;

    std::string _cat_name;
    std::string _cat_type;

    static constexpr int RATE_BASE_10000 = 10000;
    uint32_t _level{ 0 };
    //DF_LOG_DECLARE();
};

// TYPEDEF_SHARED_PTR(TransactionMetricRecorder);

};
};

#endif // MERCURY_MONITOR_TRANSACTIONMETRICRECORDER_H


/* Local Variables: */
/* tab-width: 4 */
/* indent-tabs-mode: nil */
/* coding: utf-8-unix */
/* mode: c++ */
/* End: */
// vim: set tabstop=4 expandtab fileencoding=utf-8 ff=unix ft=cpp norl :