#ifndef MERCURY_MONITOR_COMPONENT_H_
#define MERCURY_MONITOR_COMPONENT_H_

#include <string>
#include <unordered_map>
#include <functional>
#include "putil/Lock.h"
#include "butil/time.h"
#include "cat/cat.h"
#include "monitor_common.h"

namespace mercury {

class MetricRecorder;
class TransactionRecorder;

template<class T>
struct SingletonBase {
    static T& Instance() {
        static T instance;
        return instance;
    }
};

// ziqian: since Monitor is actually a global thing,
// wrap this in a singleton to ensure that names are globally unique
class MonitorComponent: public SingletonBase<MonitorComponent>{
    friend class SingletonBase<MonitorComponent>;
 public:
    typedef std::function<void(int /* latency_us */, bool /* succ */)> transaction;

    typedef std::function<void(int /* value */)> metric;

    MonitorComponent();

    ~MonitorComponent();

    transaction declare_transaction(const std::string&, const std::string&);

    metric declare_metric(const std::string&);

private:
    putil::ReadWriteLock _lock;
    std::unordered_map<std::string, TransactionRecorder*> _transaction_recorder_registry;
    std::unordered_map<std::string, MetricRecorder*>      _metric_recorder_registry;
};

}  // namespace mercury

#endif  // MERCURY_MONITOR_COMPONENT_H_
