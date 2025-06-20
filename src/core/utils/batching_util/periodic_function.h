/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     periodic_function.h
 *   \author   anduo@xiaohongshu.com
 *   \date     July 2022
 *   \version  1.0.0
 *   \brief    bthread implementation of periodic function
 */

// PeriodicFunction will periodically call the given function with a specified
// period in a background thread.  After Start() returns, the thread is
// guaranteed to have started. The destruction of the class causes the
// background thread to be destroyed as well.  Start() should not be called more
// than once.
//
// PeriodicFunction runs the function as soon as any previous run both is
// complete and was started more than "interval_micros" earlier.  Thus, runs are
// both serialized, and normally have a period of "interval_micros" if no run
// exceeds the time.
//
// Note that, if the function takes longer than two interval_micross to finish,
// then PeriodicFunction will "skip" at least one call to the function.  For
// instance, if the period is 50ms and the function starts runs at time 0 for
// 150ms, then the function will immediately start executing again at time 150,
// but there will be no function runs corresponding to times 50 or 100.  This is
// especially important to remember when using an environment with a simulated
// clock: advancing simulated time atomically over N interval_micross will not
// cause the function to be called N times.
//
// This object is thread-safe.
//
// Example:
//
//   class Foo {
//    public:
//     Foo() : periodic_function_([this]() { Bar(); },
//                               1000 /* 1000us == 1ms*/) {
//     }
//
//    private:
//     void Bar() { ... }
//
//     PeriodicFunction periodic_function_;
//   };

#ifndef MERCURY_CORE_FRAMEWORK_PERIODIC_FUNCTION_H_
#define MERCURY_CORE_FRAMEWORK_PERIODIC_FUNCTION_H_

#include "absl/synchronization/notification.h"
#include "bthread/bthread.h"
#include "leveldb/env.h"
#include "putil/Thread.h"
#include "src/core/common/common.h"
#include "src/core/framework/index_framework.h"
#include <mutex>

MERCURY_NAMESPACE_BEGIN(core);

class PeriodicFunction
{
public:
    // Provides the ability to customize several aspects of the
    // PeriodicFunction. Passed to constructor of PeriodicFunction.
    struct Options
    {
        Options() {}

        // Any standard thread options, such as stack size, should
        // be passed via "thread_options".
        // ThreadOptions thread_options;

        // Specifies the thread name prefix (see the description in class
        // Thread).
        std::string thread_name_prefix = "periodic_function";

        // The environment to use. Does not take ownership, but must remain
        // alive for as long as the PeriodicFunction exists.
        leveldb::Env *env = leveldb::Env::Default();

        // Specifies the length of sleep before the first invocation of the
        // function.
        // This can be used for adding a random jitter to avoid synchronous
        // behavior across multiple periodic functions.
        uint64_t startup_delay_micros = 0;
    };

    // Also starts the background thread which will be calling the function.
    PeriodicFunction(const std::function<void()> &function,
                     uint64_t interval_micros,
                     const Options &options = Options());
    PeriodicFunction(const PeriodicFunction &) = delete;
    void operator=(const PeriodicFunction &) = delete;

    ~PeriodicFunction();

private:
    // Notifies the background thread to stop.
    void NotifyStop();

    // (Blocking.) Loops forever calling "function_" every "interval_micros_".
    static void *RunLoop(void *);

    const std::function<void()> function_; // Actual client function
    const uint64_t interval_micros_;       // Interval between calls.
    const Options options_;

    // Used to notify the thread to stop.
    absl::Notification stop_thread_;

    bthread_t bid_;
};

MERCURY_NAMESPACE_END(core);

#endif // MERCURY_CORE_FRAMEWORK_PERIODIC_FUNCTION_H_
