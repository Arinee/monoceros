#ifndef MERCURY_MONITOR_COMMON_H_
#define MERCURY_MONITOR_COMMON_H_

// Monitor Common Define
#define MONITOR_TRANSACTION(type, name) \
    transaction_##name = MonitorComponent::Instance().declare_transaction(#type, #name); \

#define MONITOR_TRANSACTION_LOG(Func, ErrorMsg, type, name) \
    timer.start(); \
    if ( Func != 0) { \
        LOG_ERROR(ErrorMsg); \
        timer.stop(); \
        transaction_##name(timer.u_elapsed(), false); \
        return -1; \
    } \
    timer.stop(); \
    transaction_##name(timer.u_elapsed(), true); \

#define MONITOR_METRIC(name) \
    metric_##name = MonitorComponent::Instance().declare_metric(#name);

#define MONITOR_METRIC_WITH_INDEX(name, name_with_index) \
    metric_##name = MonitorComponent::Instance().declare_metric(name_with_index);

#define MONITOR_METRIC_LOG(name,value) \
    metric_##name(value); \

#endif //MERCURY_MONITOR_COMMON_H_