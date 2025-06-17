#include "monitor_component.h"
#include "bvar/bvar.h"
#include <memory>
#include "src/core/framework/index_logger.h"

DECLARE_string(cat_domain);
DECLARE_string(cat_client_config_file);
DEFINE_bool(mercury_enable_cat_monitor, false, "enable cat monitor for mercury");

namespace bvar {
DECLARE_int32(bvar_dump_interval);
DECLARE_string(bvar_dump_prefix);
DECLARE_bool(bvar_dump);
}

namespace mercury {

static const int RATE_BASE_10000 = 10000;

class TransactionRecorder {
 public:
    TransactionRecorder() : _failure_window(&_failure_recorder, bvar::FLAGS_bvar_dump_interval) {}

    int expose(const std::string& type, const std::string& name) {
        int rc = 0;
        rc = _latency_recorder.expose("transaction_" + type + "_" + name);
        if (0 != rc) {
            LOG_ERROR("expose LatencyRecorder[%s_%s] failed", type.c_str(), name.c_str());
            return rc;
        }

        rc = _failure_window.expose("transaction_" + type + "_" + name + "_failure_rate");
        if (0 != rc) {
            LOG_ERROR("expose Window[%s_%s] failed", type.c_str(), name.c_str());
            return rc;
        }

        _type = type;
        _name = name;
        if (bvar::FLAGS_bvar_dump_prefix.empty()) {
            _cat_type = type;
        } else {
            _cat_type = bvar::FLAGS_bvar_dump_prefix + "_" + type;
        }
        LOG_INFO("expose TransactionRecorder[%s_%s] successfully", type.c_str(), name.c_str());
        return 0;
    }

    void record(int latency_us, bool succ) {
        _latency_recorder << latency_us;
        _failure_recorder << (succ ? 0 : RATE_BASE_10000);

        if (FLAGS_mercury_enable_cat_monitor) {
            auto transaction = cat::Cat::Instance().NewTransaction(_cat_type, _name);
            transaction->SetStatus(succ ? "0" : "FAILED");
            transaction->SetDuration(latency_us);  // time in us
            transaction->Complete();
        }
    }

    static void dummy(int, bool) {}

 private:
    bvar::LatencyRecorder           _latency_recorder;
    bvar::IntRecorder               _failure_recorder;
    bvar::Window<bvar::IntRecorder> _failure_window;
    std::string _type;
    std::string _name;
    std::string _cat_type;
};

class MetricRecorder {
 public:
    MetricRecorder() : _window(&_int_recorder, bvar::FLAGS_bvar_dump_interval) {}
    int expose(const std::string& name) {
        int rc = _window.expose("business_metric_" + name);
        if (0 != rc) {
            LOG_ERROR("expose Window[%s] failed", name.c_str());
            return rc;
        }

        _name = name;
        if (bvar::FLAGS_bvar_dump_prefix.empty()) {
            _cat_name = name;
        } else {
            _cat_name = bvar::FLAGS_bvar_dump_prefix + "_" + name;
        }
        LOG_INFO("expose MetricRecorder[%s] successfully", name.c_str());
        return 0;
    }
    void record(int value) {
        _int_recorder << value;
        if (FLAGS_mercury_enable_cat_monitor) {
            cat::Cat::Instance().LogMetricForSum(_cat_name, value);
        }
    }

    static void dummy(int) {}

 private:
    bvar::IntRecorder               _int_recorder;
    bvar::Window<bvar::IntRecorder> _window;
    std::string _name;
    std::string _cat_name;
};

MonitorComponent::MonitorComponent() {}

MonitorComponent::~MonitorComponent() {
    putil::ScopedWriteLock lock(_lock);
    for (auto & one : _transaction_recorder_registry) {
        delete one.second;
    }
    _transaction_recorder_registry.clear();
    
    for (auto & one : _metric_recorder_registry) {
        delete one.second;
    }
    _metric_recorder_registry.clear();
}

std::function<void(int /* latency_us */, bool /* succ */)>
MonitorComponent::declare_transaction(const std::string& type, const std::string& name) {
    std::string registry_key = type + "$$" + name;
    TransactionRecorder *transaction_recorder = nullptr;

    {
        putil::ScopedReadLock lock(_lock);
        const auto iter = _transaction_recorder_registry.find(registry_key);
        if (_transaction_recorder_registry.cend() != iter) {
            transaction_recorder = iter->second;
        }
    }
    if (transaction_recorder == nullptr) {
        putil::ScopedWriteLock lock(_lock);
        const auto iter = _transaction_recorder_registry.find(registry_key);
        if (_transaction_recorder_registry.cend() != iter) {
            transaction_recorder = iter->second;
        } else {
            std::unique_ptr<TransactionRecorder> recorder(new(std::nothrow) TransactionRecorder());
            if (recorder) {
                int rc = recorder->expose(type, name);
                if (0 == rc) {
                    _transaction_recorder_registry[registry_key] = recorder.get();
                    transaction_recorder = recorder.release();
                }
            }
        }
    }

    if (nullptr == transaction_recorder) {
        return std::bind(TransactionRecorder::dummy, std::placeholders::_1, std::placeholders::_2);
    } else {
        return std::bind(&TransactionRecorder::record, transaction_recorder, std::placeholders::_1,
                         std::placeholders::_2);
    }
}

std::function<void(int /* value */)> MonitorComponent::declare_metric(const std::string& name) {
    MetricRecorder* metric_recorder = nullptr;

    {
        putil::ScopedReadLock lock(_lock);
        const auto iter = _metric_recorder_registry.find(name);
        if (_metric_recorder_registry.cend() != iter) {
            metric_recorder = iter->second;
        }
    }
    if (metric_recorder == nullptr) {
        putil::ScopedWriteLock lock(_lock);
        const auto iter = _metric_recorder_registry.find(name);
        if (_metric_recorder_registry.cend() != iter) {
            metric_recorder = iter->second;
        } else {
            std::unique_ptr<MetricRecorder> recorder(new (std::nothrow) MetricRecorder());
            if (recorder) {
                int rc = recorder->expose(name);
                if (0 == rc) {
                    _metric_recorder_registry[name] = recorder.get();
                    metric_recorder                 = recorder.release();
                }
            }
        }
    }

    if (nullptr == metric_recorder) {
        return std::bind(MetricRecorder::dummy, std::placeholders::_1);
    } else {
        return std::bind(&MetricRecorder::record, metric_recorder, std::placeholders::_1);
    }
}


}  // namespace mercury