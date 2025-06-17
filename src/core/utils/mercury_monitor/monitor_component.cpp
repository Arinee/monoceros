#include <memory>
#include "monitor_component.h"
#include "bvar/bvar.h"
#include "cat/cat.h"
#include "integer_metric_recorder.h"
#include "transaction_metric_recorder.h"
#include "event_metric_recorder.h"
#include "status_metric_recorder.h"

DECLARE_string(cat_domain);

namespace mercury {
namespace monitor {

MonitorComponent::MonitorComponent() {}

MonitorComponent::~MonitorComponent()
{
    if (!closed_) {
        Close();
    }
}

void MonitorComponent::Close()
{
    putil::ScopedWriteLock lock(_lock);
    for (auto &one : _transaction_recorder_registry) {
        delete one.second;
    }
    _transaction_recorder_registry.clear();

    for (auto &one : _metric_recorder_registry) {
        delete one.second;
    }
    _metric_recorder_registry.clear();

    for (auto &one : _event_metric_recorder_registry) {
        delete one.second;
    }
    _event_metric_recorder_registry.clear();

    for (auto &one : _status_metric_recorder_registry) {
        delete one.second;
    }
    _status_metric_recorder_registry.clear();
    closed_ = true;
}

transaction_metric_func_t
MonitorComponent::declare_transaction_metric(const std::string &name,
                                             const uint32_t transaction_level)
{
    std::string registry_key = name;
    TransactionMetricRecorder *result = nullptr;

    {
        putil::ScopedReadLock lock(_lock);
        const auto iter = _transaction_recorder_registry.find(registry_key);
        if (_transaction_recorder_registry.cend() != iter) {
            result = iter->second;
        }
    }
    if (result == nullptr) {
        putil::ScopedWriteLock lock(_lock);
        const auto iter = _transaction_recorder_registry.find(registry_key);
        if (_transaction_recorder_registry.cend() != iter) {
            result = iter->second;
        } else {
            std::unique_ptr<TransactionMetricRecorder> unique_recorder(new (
                std::nothrow) TransactionMetricRecorder(transaction_level));
            if (unique_recorder) {
                int rc = unique_recorder->expose(name);
                if (0 == rc) {
                    _transaction_recorder_registry[registry_key] =
                        unique_recorder.get();
                    result = unique_recorder.release();
                }
            }
        }
    }

    if (nullptr == result) {
        return std::bind(TransactionMetricRecorder::dummy,
                         std::placeholders::_1, std::placeholders::_2);
    } else {
        return std::bind(&TransactionMetricRecorder::record, result,
                         std::placeholders::_1, std::placeholders::_2);
    }
}

integer_metric_func_t
MonitorComponent::declare_integer_metric(const std::string &name)
{
    IntegerMetricRecorder *result = nullptr;

    {
        putil::ScopedReadLock lock(_lock);
        const auto iter = _metric_recorder_registry.find(name);
        if (_metric_recorder_registry.cend() != iter) {
            result = iter->second;
        }
    }
    if (result == nullptr) {
        putil::ScopedWriteLock lock(_lock);
        const auto iter = _metric_recorder_registry.find(name);
        if (_metric_recorder_registry.cend() != iter) {
            result = iter->second;
        } else {
            std::unique_ptr<IntegerMetricRecorder> unique_recorder(
                new (std::nothrow) IntegerMetricRecorder());
            if (unique_recorder) {
                int rc = unique_recorder->expose(name);
                if (0 == rc) {
                    _metric_recorder_registry[name] = unique_recorder.get();
                    result = unique_recorder.release();
                }
            }
        }
    }

    if (nullptr == result) {
        return std::bind(IntegerMetricRecorder::dummy, std::placeholders::_1);
    } else {
        return std::bind(&IntegerMetricRecorder::record, result,
                         std::placeholders::_1);
    }
}

event_metric_func_t
MonitorComponent::declare_event_metric(const std::string &name,
                                       const uint32_t event_level)
{
    EventMetricRecorder *result = nullptr;

    {
        putil::ScopedReadLock lock(_lock);
        const auto iter = _event_metric_recorder_registry.find(name);
        if (iter != _event_metric_recorder_registry.cend()) {
            result = iter->second;
        }
    }
    if (result == nullptr) {
        putil::ScopedWriteLock lock(_lock);
        const auto iter = _event_metric_recorder_registry.find(name);
        if (iter != _event_metric_recorder_registry.cend()) {
            result = iter->second;
        } else {
            std::unique_ptr<EventMetricRecorder> unique_recorder(
                new (std::nothrow) EventMetricRecorder(event_level));
            if (unique_recorder) {
                int rc = unique_recorder->expose(name);
                if (0 == rc) {
                    _event_metric_recorder_registry[name] =
                        unique_recorder.get();
                    result = unique_recorder.release();
                }
            }
        }
    }

    if (nullptr == result) {
        return std::bind(EventMetricRecorder::dummy, std::placeholders::_1,
                         std::placeholders::_2);
    } else {
        return std::bind(&EventMetricRecorder::record, result,
                         std::placeholders::_1, std::placeholders::_2);
    }
}

status_metric_func_t
MonitorComponent::declare_status_metric(const std::string &name)
{
    StatusMetricRecorder *result = nullptr;

    {
        putil::ScopedReadLock lock(_lock);
        const auto iter = _status_metric_recorder_registry.find(name);
        if (_status_metric_recorder_registry.cend() != iter) {
            result = iter->second;
        }
    }
    if (result == nullptr) {
        putil::ScopedWriteLock lock(_lock);
        const auto iter = _status_metric_recorder_registry.find(name);
        if (_status_metric_recorder_registry.cend() != iter) {
            result = iter->second;
        } else {
            std::unique_ptr<StatusMetricRecorder> unique_recorder(
                new (std::nothrow) StatusMetricRecorder());
            if (unique_recorder) {
                int rc = unique_recorder->expose(name);
                if (0 == rc) {
                    _status_metric_recorder_registry[name] =
                        unique_recorder.get();
                    result = unique_recorder.release();
                }
            }
        }
    }

    if (nullptr == result) {
        return std::bind(StatusMetricRecorder::dummy, std::placeholders::_1);
    } else {
        return std::bind(&StatusMetricRecorder::record, result,
                         std::placeholders::_1);
    }
}

};
};

/* Local Variables: */
/* tab-width: 4 */
/* indent-tabs-mode: nil */
/* coding: utf-8-unix */
/* mode: c++ */
/* End: */
// vim: set tabstop=4 expandtab fileencoding=utf-8 ff=unix ft=cpp norl :