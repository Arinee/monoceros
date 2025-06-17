/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     periodic_function.cc
 *   \author   anduo@xiaohongshu.com
 *   \date     August 2022
 *   \version  1.0.0
 *   \brief    bthread implementation of periodic function
 */

#include "src/core/utils/batching_util/periodic_function.h"

MERCURY_NAMESPACE_BEGIN(core);

PeriodicFunction::PeriodicFunction(const std::function<void()> &function,
                                   const uint64_t interval_micros,
                                   const Options &options)
    : function_(function), interval_micros_(interval_micros), options_(options)
{
    if (bthread_start_background(&bid_, NULL, RunLoop, this) != 0) {
        LOG_ERROR("start bthread failed.");
        // return -1;
    }
}

PeriodicFunction::~PeriodicFunction()
{
    LOG_INFO("PeriodicFunction destructor start");
    NotifyStop();
    bthread_join(bid_, NULL);
    LOG_INFO("PeriodicFunction destructor end");
}

void PeriodicFunction::NotifyStop()
{
    if (!stop_thread_.HasBeenNotified()) {
        stop_thread_.Notify();
        LOG_INFO("stop_thread_ HasBeenNotified");
    }
}

void *PeriodicFunction::RunLoop(void *periodic_func)
{
    LOG_INFO("Start PeriodicFunction RunLoop");
    PeriodicFunction *this_periodic = (PeriodicFunction *)periodic_func;
    {
        const uint64_t start = this_periodic->options_.env->NowMicros();
        if (this_periodic->options_.startup_delay_micros > 0) {
            const uint64_t deadline =
                start + this_periodic->options_.startup_delay_micros;
            bthread_usleep(deadline - start);
        }

        while (!this_periodic->stop_thread_.HasBeenNotified()) {
            const uint64_t begin = this_periodic->options_.env->NowMicros();
            this_periodic->function_();

            // Take the max() here to guard against time going backwards which
            // sometimes happens in multiproc machines.
            const uint64_t end = std::max(
                static_cast<uint64_t>(this_periodic->options_.env->NowMicros()),
                begin);

            // The deadline is relative to when the last function started.
            const uint64_t deadline = begin + this_periodic->interval_micros_;
            // We want to sleep until 'deadline'.
            if (deadline > end) {
                if (end > begin) {
                    // LOG_INFO_INTERVAL(
                    //     10000, "Reducing interval_micros from %lu to %lu",
                    //     this_periodic->interval_micros_, deadline - end);
                }
                // this_periodic->options_.env->SleepForMicroseconds(deadline -
                //                                                   end);
                bthread_usleep(deadline - end);
            } else {
                LOG_INFO("Function took %lu longer than interval_micros, so not "
                         "sleeping", end - deadline);
            }
        }
    }
    LOG_INFO("End PeriodicFunction RunLoop");

    return nullptr;
}

MERCURY_NAMESPACE_END(core);