#ifndef MERCURY_MONITOR_COMPONENT_H_
#define MERCURY_MONITOR_COMPONENT_H_

#include <functional>
#include <string>
#include <unordered_map>
#include <iostream>

#include "putil/Lock.h"
#include "cat/cat.h"
#include "singleton.h"
#include "transaction_metric_recorder.h"
#include "status_metric_recorder.h"
#include "integer_metric_recorder.h"
#include "event_metric_recorder.h"

namespace mercury {
namespace monitor {

typedef std::function<void(int /* latency_us */, bool /* succ */)>
    transaction_metric_func_t;
typedef std::function<void(long /* value */)> integer_metric_func_t;
typedef std::function<void(int /* hit_count */, int /* failure_count */)>
    event_metric_func_t;
typedef std::function<void(long /* status_value */)> status_metric_func_t;

struct TransactionMetricFunc
{
    static void dummy_metric(int, bool){};
    transaction_metric_func_t func_;
    TransactionMetricFunc(transaction_metric_func_t func) : func_(func){};
    TransactionMetricFunc() : func_(dummy_metric){};
    void operator()(int latency_us, bool succ) const
    {
        func_(latency_us, succ);
    }
};

struct IntegerMetricFunc
{
    static void dummy_metric(long){};
    integer_metric_func_t func_;
    IntegerMetricFunc(integer_metric_func_t func) : func_(func){};
    IntegerMetricFunc() : func_(dummy_metric){};
    void operator()(long value) const
    {
        func_(value);
    }
};

struct EventMetricFunc
{
    static void dummy_metric(int, int){};
    event_metric_func_t func_;
    EventMetricFunc(event_metric_func_t func) : func_(func){};
    EventMetricFunc() : func_(dummy_metric){};
    void operator()(int hit_count, int failure_count) const
    {
        func_(hit_count, failure_count);
    }
};

struct StatusMetricFunc
{
    static void dummy_metric(long){};
    status_metric_func_t func_;
    StatusMetricFunc(status_metric_func_t func) : func_(func){};
    StatusMetricFunc() : func_(dummy_metric){};
    void operator()(long status_value) const
    {
        func_(status_value);
    }
};

// NB: since Monitor is actually a global thing,
// wrap this in a singleton to ensure that names are globally unique
class MonitorComponent : public util::Singleton<MonitorComponent>
{
    friend class util::Singleton<MonitorComponent>;

public:
    MonitorComponent();
    ~MonitorComponent();
    void Close();
    // 以下方法对同一组参数的调用是可重入的
    // declare的过程有可能很耗时但保证是thread-safe的，返回值的调用也是thread-safe的

    // 用于上报一个过程的失败率、耗时
    transaction_metric_func_t
    declare_transaction_metric(const std::string &name,
                               const uint32_t transaction_level = 0);
    // 用于上报某个指标的值(整型值)
    integer_metric_func_t declare_integer_metric(const std::string &name);
    // 用于上报某个事件, 可以包括失败次数
    event_metric_func_t declare_event_metric(const std::string &name,
                                             const uint32_t event_level = 0);
    // 用于上报某个状态值
    status_metric_func_t declare_status_metric(const std::string &name);

private:
    putil::ReadWriteLock _lock;
    std::unordered_map<std::string, TransactionMetricRecorder *>
        _transaction_recorder_registry;
    std::unordered_map<std::string, IntegerMetricRecorder *>
        _metric_recorder_registry;
    std::unordered_map<std::string, EventMetricRecorder *>
        _event_metric_recorder_registry;
    std::unordered_map<std::string, StatusMetricRecorder *>
        _status_metric_recorder_registry;
    bool closed_ = false;
};

};
};

#endif // MERCURY_MONITOR_COMPONENT_H_

/* Local Variables: */
/* tab-width: 4 */
/* indent-tabs-mode: nil */
/* coding: utf-8-unix */
/* mode: c++ */
/* End: */
// vim: set tabstop=4 expandtab fileencoding=utf-8 ff=unix ft=cpp norl :