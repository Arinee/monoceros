/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     shared_batch_scheduler.h
 *   \author   anduo@xiaohongshu.com
 *   \date     July 2022
 *   \version  1.0.0
 *   \brief    bthread implementation of shared batch scheduler
 */

#ifndef MERCURY_CORE_FRAMEWORK_SHARED_BATCH_SCHEDULER_H_
#define MERCURY_CORE_FRAMEWORK_SHARED_BATCH_SCHEDULER_H_

#include "src/core/common/common.h"
#include "src/core/utils/batching_util/batch_scheduler.h"
#include "src/core/utils/batching_util/periodic_function.h"
#include "src/core/utils/batching_util/batch_queue.h"
#include <list>

MERCURY_NAMESPACE_BEGIN(core);
// A batch scheduler for server instances that service multiple request types
// (e.g. multiple machine-learned models, or multiple versions of a model served
// concurrently), or even multiple distinct tasks for a given request. The
// scheduler multiplexes batches of different kinds of tasks onto a fixed-size
// thread pool (each batch contains tasks of a single type), in a carefully
// controlled manner. A common configuration is to set the number of threads
// equal to the number of hardware accelerator units, in which case the
// scheduler takes care of multiplexing the task types onto the shared hardware,
// in a manner that is both fair and efficient.
//
// Semantically, SharedBatchScheduler behaves like having N instances of
// BasicBatchScheduler (see basic_batch_scheduler.h), one per task type. The
// difference is that under the covers there is a single shared thread pool,
// instead of N independent ones, with their sharing deliberately coordinated.
//
// SharedBatchScheduler does not implement the BatchScheduler API; rather, it
// presents an abstraction of "queues", where each queue corresponds to one type
// of task. Tasks submitted to a given queue are placed in their own batches,
// and cannot be mixed with other tasks. Queues can be added and deleted
// dynamically, to accommodate e.g. versions of a model being brought up and
// down over the lifetime of a server.
//
// The batch thread pool round-robins through the queues, running one batch
// from a queue and then moving to the next queue. Each queue behaves like a
// BasicBatchScheduler instance, in the sense that it has maximum batch size and
// timeout parameters, which govern when a batch is eligible to be processed.
//
// Each queue is independently configured with a maximum size (in terms of the
// maximum number of batches worth of enqueued tasks). For online serving, it is
// recommended that the queue sizes be configured such that the sum of the sizes
// of the active queues roughly equal the number of batch threads. (The idea is
// that if all threads become available at roughly the same time, there will be
// enough enqueued work for them to take on, but no more.)
//
// If queue sizes are configured in the manner suggested above, the maximum time
// a task can spend in a queue before being placed in a batch and assigned to a
// thread for processing, is the greater of:
//  - the maximum time to process one batch of tasks from any active queue
//  - the configured timeout parameter for the task's queue (which can be 0)
//
// For bulk processing jobs and throughput-oriented benchmarks, you may want to
// set the maximum queue size to a large value.
//
// TODO(b/26539183): Support queue servicing policies other than round-robin.
// E.g. let each queue specify a "share" (an int >= 1), so e.g. with queues A
// and B having shares 1 and 2 respectively, the servicing pattern is ABBABB...
//
//
// PERFORMANCE TUNING: See README.md.
//
template <typename TaskType>
class SharedBatchScheduler
{
public:
    struct Options
    {
        std::string thread_pool_name = { "batch_threads" };
        int num_batch_threads = 1;
        leveldb::Env *env = leveldb::Env::Default();
        uint64_t periodic_interval_micros = 0;
    };

    explicit SharedBatchScheduler(const Options &options);
    SharedBatchScheduler(const SharedBatchScheduler &) = delete;
    void operator=(const SharedBatchScheduler &) = delete;

    ~SharedBatchScheduler();

    // Adds a queue to which tasks may be submitted. The returned queue
    // implements the BatchScheduler API. Each queue has its own set of
    // scheduling options, and its own callback to process batches of tasks
    // submitted to the queue.
    //
    // The returned queue's destructor blocks until all tasks submitted to it
    // have been processed.
    std::shared_ptr<BatchQueue<TaskType>>
    AddQueue(const typename BatchQueue<TaskType>::BatchQueueOptions &options,
             std::function<void(std::shared_ptr<Batch<TaskType>>)>
                 process_batch_callback);

    // 基础抽象类函数
    // virtual Bool Schedule(std::unique_ptr<TaskType> *task) override;
    // virtual size_t NumEnqueuedTasks() const override;
    // virtual size_t SchedulingCapacity() const override;
    // virtual size_t max_task_size() const override;

private:
    // The code executed in 'batch_threads_'. Obtains a batch to process from
    // the queue pointed to by 'next_queue_to_schedule_', and processes it. If
    // that queue declines to provide a batch to process, moves onto the next
    // queue. If no queues provide a batch to process, just sleeps briefly and
    // exits.
    void ThreadLogic();

    const Options options_;

    bthread::Mutex mu_;

    using QueueList = std::list<std::shared_ptr<BatchQueue<TaskType>>>;

    QueueList queues_;

    typename QueueList::iterator next_queue_to_schedule_;

    // Threads that process batches obtained from the queues.
    std::vector<std::unique_ptr<PeriodicFunction>> batch_threads_;

    struct ProcessBatchMessage
    {
        std::shared_ptr<Batch<TaskType>> batch_to_process_;
        BatchQueue<TaskType> *queue_for_batch_;
        ProcessBatchMessage(std::unique_ptr<Batch<TaskType>> batch_to_process,
                            BatchQueue<TaskType> *queue_for_batch)
            : batch_to_process_(std::move(batch_to_process)),
              queue_for_batch_(queue_for_batch){};
    };

    static void *BthreadProcessBatch(void *);
};

template <typename TaskType>
SharedBatchScheduler<TaskType>::SharedBatchScheduler(const Options &options)
    : options_(options), next_queue_to_schedule_(queues_.end())
{
    // Kick off the batch threads.
    PeriodicFunction::Options periodic_fn_options;
    periodic_fn_options.thread_name_prefix = options.thread_pool_name + "_";
    for (int i = 0; i < options.num_batch_threads; ++i) {
        std::unique_ptr<PeriodicFunction> thread(new PeriodicFunction(
            [this] { this->ThreadLogic(); }, options.periodic_interval_micros,
            periodic_fn_options));
        batch_threads_.push_back(std::move(thread));
    }
}

// template <typename TaskType>
// Bool SharedBatchScheduler<TaskType>::Create(
//     const Options &options,
//     std::shared_ptr<SharedBatchScheduler<TaskType>> *scheduler)
// {
//     if (options.num_batch_threads < 1) {
//         LOG_ERROR("num_batch_threads must be positive; was %d",
//                   options.num_batch_threads);
//         return false;
//     }
//     scheduler->reset(new SharedBatchScheduler<TaskType>(options));
//     return false;
// }


template <typename TaskType>
SharedBatchScheduler<TaskType>::~SharedBatchScheduler()
{
    // Wait until the batch threads finish clearing out and deleting the closed
    // queues.
    LOG_INFO("SharedBatchScheduler desturctor start");
    for (;;) {
        {
            std::lock_guard<bthread::Mutex> l(mu_);
            if (queues_.empty()) {
                break;
            }
        }
        const uint64_t kSleepTimeMicros = 100;
        options_.env->SleepForMicroseconds(kSleepTimeMicros);
    }
    // Delete the batch threads before allowing state the threads may access
    // (e.g. 'mu_') to be deleted.
    batch_threads_.clear();
    LOG_INFO("SharedBatchScheduler desturctor complete");
}

template <typename TaskType>
std::shared_ptr<BatchQueue<TaskType>> SharedBatchScheduler<TaskType>::AddQueue(
    const typename BatchQueue<TaskType>::BatchQueueOptions &options,
    std::function<void(std::shared_ptr<Batch<TaskType>>)>
        process_batch_callback)
{
    if (options.max_batch_size == 0) {
        LOG_ERROR("max_batch_size must be positive; was %lu",
                  options.max_batch_size);
        return nullptr;
    }
    if (options.batch_timeout_micros < 0) {
        LOG_ERROR("batch_timeout_micros must be non-negative; was %lu",
                  options.batch_timeout_micros);
        return nullptr;
    }
    if (options.max_enqueued_batches < 0) {
        LOG_ERROR("max_enqueued_batches must be non-negative; was %lu",
                  options.max_enqueued_batches);
        return nullptr;
    }

    auto schedulable_batch_callback = [this] {
        // lock_guard<bthread::Mutex> l(mu_);
        // schedulable_batch_cv_.notify_one();
    };
    auto internal_queue = std::shared_ptr<BatchQueue<TaskType>>(
        new BatchQueue<TaskType>(options, options_.env, process_batch_callback,
                                 schedulable_batch_callback));
    {
        std::lock_guard<bthread::Mutex> l(mu_);
        queues_.push_back(internal_queue);
        if (next_queue_to_schedule_ == queues_.end()) {
            next_queue_to_schedule_ = queues_.begin();
        }
    }

    return internal_queue;
}

template <typename TaskType>
void SharedBatchScheduler<TaskType>::ThreadLogic()
{
    // A batch to process next (or nullptr if no work to do).
    std::unique_ptr<Batch<TaskType>> batch_to_process;
    // The queue with which 'batch_to_process' is associated.
    BatchQueue<TaskType> *queue_for_batch = nullptr;
    {
        std::lock_guard<bthread::Mutex> l(mu_);

        const int num_queues = queues_.size();
        for (int num_queues_tried = 0;
             batch_to_process == nullptr && num_queues_tried < num_queues;
             ++num_queues_tried) {
            // if (next_queue_to_schedule_ == queues_.end()) {
            //     LOG_ERROR("next_queue_to_schedule_ error");
            //     return;
            // }

            // If a closed queue responds to ScheduleBatch() with nullptr, the
            // queue will never yield any further batches so we can drop it. To
            // avoid a race, we take a snapshot of the queue's closedness state
            // *before* calling ScheduleBatch().
            const bool queue_closed = (*next_queue_to_schedule_)->closed();

            // Ask '*next_queue_to_schedule_' if it wants us to process a batch.
            batch_to_process = (*next_queue_to_schedule_)->ScheduleBatch();
            if (batch_to_process != nullptr) {
                queue_for_batch = next_queue_to_schedule_->get();
            }

            // Advance 'next_queue_to_schedule_'.
            if (queue_closed && (*next_queue_to_schedule_)->IsEmpty() &&
                batch_to_process == nullptr) {
                // We've encountered a closed queue with no work to do. Drop it.
                next_queue_to_schedule_ =
                    queues_.erase(next_queue_to_schedule_);
            } else {
                ++next_queue_to_schedule_;
            }
            if (next_queue_to_schedule_ == queues_.end() && !queues_.empty()) {
                // We've hit the end. Wrap to the first queue.
                next_queue_to_schedule_ = queues_.begin();
            }
        }

        if (batch_to_process == nullptr) {
            // 变更为不等待，因为只有一个线程在轮询
            // We couldn't find any work to do. Wait until a new batch becomes
            // schedulable, or some time has elapsed, before checking again.
            // const int64 kTimeoutMillis = 1;  // The smallest accepted granule
            // of time. WaitForMilliseconds(&l, &schedulable_batch_cv_,
            // kTimeoutMillis);
            return;
        }

        if (batch_to_process->num_tasks() != 0) {
            ProcessBatchMessage *msg = new ProcessBatchMessage(
                std::move(batch_to_process), queue_for_batch);

            bthread_t bid;
            if (bthread_start_background(&bid, NULL, BthreadProcessBatch,
                                         msg) != 0) {
                LOG_ERROR("start bthread failed.");
                return;
            }
        }
    }
    return;
}

template <typename TaskType>
void *SharedBatchScheduler<TaskType>::BthreadProcessBatch(void *msg)
{
    auto msg_ = (ProcessBatchMessage *)msg;
    msg_->batch_to_process_->SetTaskStartProcessTime(
        msg_->queue_for_batch_->env_->NowMicros());
    msg_->queue_for_batch_->ProcessBatch(msg_->batch_to_process_);
    delete msg_;
    return nullptr;
}

MERCURY_NAMESPACE_END(core);

#endif // MERCURY_CORE_FRAMEWORK_SHARED_BATCH_SCHEDULER_H_
