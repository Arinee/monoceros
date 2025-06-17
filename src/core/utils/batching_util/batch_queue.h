/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     batch_queue.h
 *   \author   anduo@xiaohongshu.com
 *   \date     August 2022
 *   \version  1.0.0
 *   \brief    bthread implementation of batch queue
 */

#ifndef MERCURY_CORE_FRAMEWORK_BATCH_QUEUE_H_
#define MERCURY_CORE_FRAMEWORK_BATCH_QUEUE_H_

#include "src/core/utils/batching_util/batch_scheduler.h"

MERCURY_NAMESPACE_BEGIN(core);

template <typename TaskType>
class BatchQueue
{
public:
    struct BatchQueueOptions
    {
        size_t max_batch_size = 16;
        uint64_t batch_timeout_micros = 0;
        size_t max_enqueued_batches = 1000;
        uint32_t max_batch_concurrency_num = 30;
    };

    using ProcessBatchCallback =
        std::function<void(std::shared_ptr<Batch<TaskType>>)>;
    using SchedulableBatchCallback = std::function<void()>;
    BatchQueue(const BatchQueueOptions &options, leveldb::Env *env,
               ProcessBatchCallback process_batch_callback,
               SchedulableBatchCallback schdulable_batch_callback);
    BatchQueue(const BatchQueue &) = delete;
    void operator=(const BatchQueue &) = delete;

    // Illegal to destruct unless the queue is empty.
    ~BatchQueue();

    // Submits a task to the queue, with the same semantics as
    // BatchScheduler::Schedule().
    bool Schedule(std::shared_ptr<TaskType> task);

    // Returns the number of enqueued tasks, with the same semantics as
    // BatchScheduler::NumEnqueuedTasks().
    size_t NumEnqueuedTasks() const;

    // Returns the queue capacity, with the same semantics as
    // BatchScheduler::SchedulingCapacity().
    size_t SchedulingCapacity() const;

    // Returns the maximum allowed size of tasks submitted to the queue.
    size_t max_task_size() const
    {
        return options_.max_batch_size;
    }

    // Called by a thread that is ready to process a batch, to request one from
    // this queue. Either returns a batch that is ready to be processed, or
    // nullptr if the queue declines to schedule a batch at this time. If it
    // returns a batch, the batch is guaranteed to be closed.
    std::unique_ptr<Batch<TaskType>> ScheduleBatch();

    // Processes a batch that has been returned earlier by ScheduleBatch().
    void ProcessBatch(std::shared_ptr<Batch<TaskType>> batch);

    // Determines whether the queue is empty, i.e. has no tasks waiting or being
    // processed.
    bool IsEmpty() const;

    // Marks the queue closed, and waits until it is empty.
    void CloseAndWaitUntilEmpty();

    bool closed() const
    {
        // std::lock_guard<bthread::Mutex> l(mu_);
        return closed_;
    }

    uint32_t FinishOneBatch()
    {
        std::lock_guard<bthread::Mutex> l(mu_);
        return --num_batch_concurrency_;
    }

public:
    // The environment to use.
    leveldb::Env *env_;

private:
    // Same as IsEmpty(), but assumes the caller already holds a lock on 'mu_'.
    bool IsEmptyInternal() const;

    // Closes the open batch residing at the back of 'batches_', and inserts a
    // fresh open batch behind it.
    void StartNewBatch();

    // Determines whether the open batch residing at the back of 'batches_' is
    // currently schedulable.
    bool IsOpenBatchSchedulable() const;

    const BatchQueueOptions options_;

    // A callback invoked to processes a batch of work units. Always invoked
    // from a batch thread.
    ProcessBatchCallback process_batch_callback_;

    // A callback invoked to notify the scheduler that a new batch has become
    // schedulable.
    SchedulableBatchCallback schedulable_batch_callback_;

    mutable bthread::Mutex mu_;

    // Whether this queue can accept new tasks. This variable is monotonic: it
    // starts as false, and then at some point gets set to true and remains true
    // for the duration of this object's life.
    bool closed_ = false;

    // The enqueued batches. See the invariants in the class comments above.
    std::deque<std::unique_ptr<Batch<TaskType>>> batches_;

    // The time at which the first task was added to the open (back-most) batch
    // in 'batches_'. Valid iff that batch contains at least one task.
    uint64_t open_batch_start_time_micros_;

    // Whether this queue contains a batch that is eligible to be scheduled.
    // Used to keep track of when to call 'schedulable_batch_callback_'.
    bool schedulable_batch_ = false;

    // The number of batches currently being processed by batch threads.
    // Incremented in ScheduleBatch() and decremented in ProcessBatch().
    // change to use atomic
    std::atomic<int> num_batches_being_processed_{ 0 };

    // The number of batches currently being processd by gpu (already submitted
    // by batch threads). Incremented in ScheduleBatch() and decremented in gpu
    // searcher callback.
    uint32_t num_batch_concurrency_ = 0;

    // Used by CloseAndWaitUntilEmpty() to wait until the queue is empty, for
    // the case in which the queue is not empty when CloseAndWaitUntilEmpty()
    // starts. When ProcessBatch() dequeues the last batch and makes the queue
    // empty, if 'empty_notification_' is non-null it calls
    // 'empty_notification_->Notify()'.
    absl::Notification *empty_notification_ = nullptr;

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

//////////
// Implementation details follow. API users need not read.

// A task queue for SharedBatchScheduler. Accepts tasks and accumulates them
// into batches, and dispenses those batches to be processed via a "pull"
// interface. The queue's behavior is governed by maximum batch size, timeout
// and maximum queue length parameters; see their documentation in
// SharedBatchScheduler.
//
// The queue is implemented as a deque of batches, with these invariants:
//  - The number of batches is between 1 and 'options_.max_enqueued_batches'.
//  - The back-most batch is open; the rest are closed.
//
// Submitted tasks are added to the open batch. If that batch doesn't have room
// but the queue isn't full, then that batch is closed and a new open batch is
// started.
//
// Batch pull requests are handled by dequeuing the front-most batch if it is
// closed. If the front-most batch is open (i.e. the queue contains only one
// batch) and has reached the timeout, it is immediately closed and returned;
// otherwise no batch is returned for the request.

template <typename TaskType>
BatchQueue<TaskType>::BatchQueue(
    const BatchQueueOptions &options, leveldb::Env *env,
    ProcessBatchCallback process_batch_callback,
    SchedulableBatchCallback schedulable_batch_callback)
    : options_(options), env_(env),
      process_batch_callback_(process_batch_callback),
      schedulable_batch_callback_(schedulable_batch_callback)
{
    // Create an initial, open batch.
    batches_.emplace_back(new Batch<TaskType>);
}

template <typename TaskType>
BatchQueue<TaskType>::~BatchQueue()
{
    LOG_INFO("BatchQueue destructor starts");
    std::lock_guard<bthread::Mutex> l(mu_);
    if (!IsEmptyInternal()) {
        LOG_ERROR("check empty internal failed");
        return;
    }

    // Close the (empty) open batch, so its destructor doesn't block.
    batches_.back()->Close();
    LOG_INFO("BatchQueue destructor ends");
}

template <typename TaskType>
bool BatchQueue<TaskType>::Schedule(std::shared_ptr<TaskType> task)
{
    // std::cout << "Enter scheduler" << std::endl;
    if (task->size() > options_.max_batch_size) {
        LOG_ERROR("Task size %lu is larger than maximum batch size %lu",
                  task->size(), options_.max_batch_size);
        return false;
    }

    // bool notify_of_schedulable_batch = false;
    {
        std::lock_guard<bthread::Mutex> l(mu_);

        if (closed_) {
            LOG_ERROR("batch queue is closed!");
            return false;
        }
        if (batches_.back()->size() + task->size() > options_.max_batch_size) {
            if (batches_.size() >= options_.max_enqueued_batches) {
                LOG_ERROR("The batch scheduling queue to which this task was "
                          "submitted is full");
                return false;
            }
            StartNewBatch();
        }
        if (batches_.back()->empty()) {
            open_batch_start_time_micros_ = env_->NowMicros();
        }
        batches_.back()->AddTask(task, env_->NowMicros());

        if (!schedulable_batch_) {
            if (batches_.size() > 1 || IsOpenBatchSchedulable()) {
                schedulable_batch_ = true;
                // notify_of_schedulable_batch = true;
            }
        }

        if (batches_.size() >= 2) {
            // There is at least one closed batch that is ready to be scheduled.
            // Process immediately, in case periodic thread is slow
            ++num_batches_being_processed_;
            std::unique_ptr<Batch<TaskType>> batch_to_processs =
                std::move(batches_.front());
            batches_.pop_front();

            if (batch_to_processs->num_tasks() != 0) {
                ProcessBatchMessage *msg =
                    new ProcessBatchMessage(std::move(batch_to_processs), this);

                bthread_t bid;
                if (bthread_start_background(&bid, NULL, BthreadProcessBatch,
                                             msg) != 0) {
                    LOG_ERROR("start bthread failed.");
                    return true;
                }
            }
        }
    }

    // if (notify_of_schedulable_batch) {
    //  schedulable_batch_callback_();
    //}
    // std::cout << "Exit scheduler" << std::endl;

    return true;
}

template <typename TaskType>
void *BatchQueue<TaskType>::BthreadProcessBatch(void *msg)
{
    auto msg_ = (ProcessBatchMessage *)msg;
    msg_->batch_to_process_->SetTaskStartProcessTime(
        msg_->queue_for_batch_->env_->NowMicros());
    msg_->queue_for_batch_->ProcessBatch(msg_->batch_to_process_);
    delete msg_;
    return nullptr;
}

template <typename TaskType>
size_t BatchQueue<TaskType>::NumEnqueuedTasks() const
{
    std::lock_guard<bthread::Mutex> l(mu_);
    size_t num_enqueued_tasks = 0;
    for (const auto &batch : batches_) {
        num_enqueued_tasks += batch->num_tasks();
    }
    return num_enqueued_tasks;
}

template <typename TaskType>
size_t BatchQueue<TaskType>::SchedulingCapacity() const
{
    std::lock_guard<bthread::Mutex> l(mu_);
    const int num_new_batches_schedulable =
        options_.max_enqueued_batches - batches_.size();
    const int open_batch_capacity =
        options_.max_batch_size - batches_.back()->size();
    return (num_new_batches_schedulable * options_.max_batch_size) +
           open_batch_capacity;
}

template <typename TaskType>
std::unique_ptr<Batch<TaskType>> BatchQueue<TaskType>::ScheduleBatch()
{
    // The batch to schedule, which we may populate below. (If left as nullptr,
    // that means we are electing not to schedule a batch at this time.)
    std::unique_ptr<Batch<TaskType>> batch_to_schedule;

    {
        std::lock_guard<bthread::Mutex> l(mu_);

        // Consider closing the open batch at this time, to schedule it.
        if (batches_.size() == 1 && IsOpenBatchSchedulable()) {
            StartNewBatch();
            open_batch_start_time_micros_ = env_->NowMicros();
        }

        // if (num_batch_concurrency_ >= options_.max_batch_concurrency_num) {
        //     return batch_to_schedule;
        // }

        if (batches_.size() >= 2) {
            // There is at least one closed batch that is ready to be scheduled.
            num_batches_being_processed_++;
            batch_to_schedule = std::move(batches_.front());
            // if (batch_to_schedule->num_tasks() != 0) {
            //     ++num_batch_concurrency_;
            // }
            batches_.pop_front();
        } else {
            schedulable_batch_ = false;
        }
    }

    return batch_to_schedule;
}

template <typename TaskType>
void BatchQueue<TaskType>::ProcessBatch(std::shared_ptr<Batch<TaskType>> batch)
{
    // std::cout << "Process Batch: " << batch->size() << std::endl;
    process_batch_callback_(batch);

    num_batches_being_processed_--;
}

template <typename TaskType>
bool BatchQueue<TaskType>::IsEmpty() const
{
    // std::lock_guard<bthread::Mutex> l(mu_);
    return IsEmptyInternal();
}

template <typename TaskType>
void BatchQueue<TaskType>::CloseAndWaitUntilEmpty()
{
    LOG_INFO("BatchQueue CloseAndWaitUntilEmpty starts");
    while (!IsEmptyInternal()) {
        bthread_usleep(1000);
    }
    closed_ = true;
    LOG_INFO("BatchQueue CloseAndWaitUntilEmpty ends");
}

template <typename TaskType>
bool BatchQueue<TaskType>::IsEmptyInternal() const
{
    return num_batches_being_processed_ == 0 && batches_.size() == 1 &&
           batches_.back()->empty();
}

template <typename TaskType>
void BatchQueue<TaskType>::StartNewBatch()
{
    batches_.back()->Close();
    batches_.emplace_back(new Batch<TaskType>);
}

template <typename TaskType>
bool BatchQueue<TaskType>::IsOpenBatchSchedulable() const
{
    Batch<TaskType> *open_batch = batches_.back().get();
    if (open_batch->empty()) {
        return false;
    }
    return closed_ || open_batch->size() >= options_.max_batch_size ||
           env_->NowMicros() >=
               open_batch_start_time_micros_ + options_.batch_timeout_micros;
}


MERCURY_NAMESPACE_END(core);

#endif // MERCURY_CORE_FRAMEWORK_BATCH_QUEUE_H_