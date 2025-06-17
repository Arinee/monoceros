#ifndef MERCURY_MONITOR_MACROS_H_
#define MERCURY_MONITOR_MACROS_H_

#include "monitor_component.h"

// NB: Helper Macro For Transaction Metric
#define DF_GET_TRANSACTION_METRIC(metric_name)                                 \
    mercury::monitor::MonitorComponent::GetInstance()                     \
        ->declare_transaction_metric(metric_name);

#define DF_GET_TRANSACTION_METRIC_WITH_LEVEL(metric_name, metric_level)        \
    mercury::monitor::MonitorComponent::GetInstance()                     \
        ->declare_transaction_metric(metric_name, metric_level);

// NB: Helper Macro For Integer Metric
#define DF_GET_INTEGER_METRIC(metric_name)                                     \
    mercury::monitor::MonitorComponent::GetInstance()                     \
        ->declare_integer_metric(metric_name);

// NB: Helper Macro For Status Metric
#define DF_GET_STATUS_METRIC(metric_name)                                      \
    mercury::monitor::MonitorComponent::GetInstance()                     \
        ->declare_status_metric(metric_name);

// NB: Helper Macro For Event Metric
#define DF_GET_EVENT_METRIC(metric_name)                                       \
    mercury::monitor::MonitorComponent::GetInstance()                     \
        ->declare_event_metric(metric_name);

#define DF_GET_EVENT_METRIC_WITH_LEVEL(metric_name, metric_level)              \
    mercury::monitor::MonitorComponent::GetInstance()                     \
        ->declare_event_metric(metric_name, metric_level);

#define DF_INTEGER_METRIC_LOG(metric_op, value) metric_op(value);


#endif // MERCURY_MONITOR_MACROS_H_