/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     batch_scheduler.h
 *   \author   anduo@xiaohongshu.com
 *   \date     July 2022
 *   \version  1.0.0
 *   \brief    Abstractions of BatchTask, Batch, BatchScheduler
 */

// Abstractions for processing small tasks in a batched fashion, to reduce
// processing times and costs that can be amortized across multiple tasks.
//
// The core class is BatchScheduler, which groups tasks into batches.
//
// BatchScheduler encapsulates logic for aggregating multiple tasks into a
// batch, and kicking off processing of a batch on a thread pool it manages.
//
// This file defines an abstract BatchScheduler class.

#ifndef MERCURY_CORE_FRAMEWORK_BATCH_SCHEDULER_H_
#define MERCURY_CORE_FRAMEWORK_BATCH_SCHEDULER_H_

#include "src/core/common/common.h"
#include "src/core/framework/index_framework.h"
#include "absl/synchronization/notification.h"
#include "leveldb/env.h"
#include "bthread/bthread.h"
#include <mutex>

MERCURY_NAMESPACE_BEGIN(core);

// The abstract superclass for a unit of work to be done as part of a batch.
//
// An implementing subclass typically contains (or points to):
//  (a) input data;
//  (b) a thread-safe completion signal (e.g. a Notification);
//  (c) a place to store the outcome (success, or some error), upon completion;
//  (d) a place to store the output data, upon success.
//
// Items (b), (c) and (d) are typically non-owned pointers to data homed
// elsewhere, because a task's ownership gets transferred to a BatchScheduler
// (see below) and it may be deleted as soon as it is done executing.
class BatchTask
{
public:
    virtual ~BatchTask() = default;

    // Returns the size of the task, in terms of how much it contributes to the
    // size of a batch. (A batch's size is the sum of its task sizes.)
    virtual size_t size() const = 0;
};

class FakeBatchTask : public BatchTask
{
public:
    FakeBatchTask(size_t size) : size_(size){};
    FakeBatchTask() = delete;

    virtual size_t size() const override
    {
        return size_;
    }

private:
    size_t size_;
};

// A thread-safe collection of BatchTasks, to be executed together in some
// fashion.
//
// At a given time, a batch is either "open" or "closed": an open batch can
// accept new tasks; a closed one cannot. A batch is monotonic: initially it is
// open and tasks can be added to it; then it is closed and its set of tasks
// remains fixed for the remainder of its life. A closed batch cannot be re-
// opened. Tasks can never be removed from a batch.
//
// Type parameter TaskType must be a subclass of BatchTask.
template <typename TaskType>
class Batch
{
public:
    Batch();
    Batch(const Batch &) = delete;
    void operator=(const Batch &) = delete;
    virtual ~Batch(); // Blocks until the batch is closed.

    // Appends 'task' to the batch. After calling AddTask(), the newly-added
    // task can be accessed via task(num_tasks()-1) or
    // mutable_task(num_tasks()-1). Dies if the batch is closed.
    // 新增：AddTask时，需要获取这个Batch的cond和mutex
    void AddTask(std::shared_ptr<TaskType> task, uint64_t task_add_time);

    // Removes the most recently added task. Returns nullptr if the batch is
    // empty.
    std::shared_ptr<TaskType> RemoveTask();

    // Returns the number of tasks in the batch.
    int num_tasks() const;

    // Returns true iff the batch contains 0 tasks.
    bool empty() const;

    // Returns a reference to the ith task (in terms of insertion order).
    const TaskType &task(int i) const;

    // Returns a pointer to the ith task (in terms of insertion order).
    TaskType *mutable_task(int i);

    // Returns the sum of the task sizes.
    size_t size() const;

    // Returns true iff the batch is currently closed.
    bool IsClosed() const;

    // Blocks until the batch is closed.
    void WaitUntilClosed() const;

    // Marks the batch as closed. Dies if called more than once.
    void Close();

    // bthread_cond_t *GetBthreadCond();
    // bthread_mutex_t *GetBthreadMutex();
    void SetTaskStartProcessTime(uint64_t task_start_process_time);

public:
    // record for monitor
    std::vector<uint64_t> task_add_time_;
    uint64_t task_start_process_time_;

private:
    mutable bthread::Mutex mu_;

    // The tasks in the batch.
    std::vector<std::shared_ptr<TaskType>> tasks_;

    // The sum of the sizes of the tasks in 'tasks_'.
    size_t size_ = 0;

    // Whether the batch has been closed.
    absl::Notification closed_;
};

// An abstract batch scheduler class. Collects individual tasks into batches,
// and processes each batch on a pool of "batch threads" that it manages. The
// actual logic for processing a batch is accomplished via a callback.
//
// Type parameter TaskType must be a subclass of BatchTask.
template <typename TaskType>
class BatchScheduler
{
public:
    virtual ~BatchScheduler() = default;

    // Submits a task to be processed as part of a batch.
    //
    // Ownership of '*task' is transferred to the callee iff the method returns
    // Status::OK. In that case, '*task' is left as nullptr. Otherwise, '*task'
    // is left as-is.
    //
    // If no batch processing capacity is available to process this task at the
    // present time, and any task queue maintained by the implementing subclass
    // is full, this method returns an UNAVAILABLE error code. The client may
    // retry later.
    //
    // Other problems, such as the task size being larger than the maximum batch
    // size, yield other, permanent error types.
    //
    // In all cases, this method returns "quickly" without blocking for any
    // substantial amount of time. If the method returns Status::OK, the task is
    // processed asynchronously, and any errors that occur during the processing
    // of the batch that includes the task can be reported to 'task'.
    // Status -> Bool
    virtual bool Schedule(std::shared_ptr<TaskType> task) = 0;

    // Returns the number of tasks that have been scheduled (i.e. accepted by
    // Schedule()), but have yet to be handed to a thread for execution as part
    // of a batch. Note that this returns the number of tasks, not the aggregate
    // task size (so if there is one task of size 3 and one task of size 5, this
    // method returns 2 rather than 8).
    virtual size_t NumEnqueuedTasks() const = 0;

    // Returns a guaranteed number of size 1 tasks that can be Schedule()d
    // without getting an UNAVAILABLE error. In a typical implementation,
    // returns the available space on a queue.
    //
    // There are two important caveats:
    //  1. The guarantee does not extend to varying-size tasks due to possible
    //     internal fragmentation of batches.
    //  2. The guarantee only holds in a single-thread environment or critical
    //     section, i.e. if an intervening thread cannot call Schedule().
    //
    // This method is useful for monitoring, or for guaranteeing a future slot
    // in the schedule (but being mindful about the caveats listed above).
    virtual size_t SchedulingCapacity() const = 0;

    // Returns the maximum allowed size of tasks submitted to the scheduler.
    // (This is typically equal to a configured maximum batch size.)
    virtual size_t max_task_size() const = 0;

    virtual uint32_t FinishOneBatch() const = 0;
};

//////////
// Implementation details follow. API users need not read.
template <typename TaskType>
Batch<TaskType>::Batch()
{
    // std::cout << "batch initialized: " << (void *)&cond_ << std::endl;
    // bthread_cond_init(&cond_, NULL);
    // bthread_mutex_init(&mutex_, NULL);
}

template <typename TaskType>
Batch<TaskType>::~Batch()
{
    WaitUntilClosed();
}

template <typename TaskType>
void Batch<TaskType>::AddTask(std::shared_ptr<TaskType> task,
                              uint64_t task_add_time)
{
    DCHECK(!IsClosed());
    {
        std::lock_guard<bthread::Mutex> l(mu_);
        size_ += task->size();
        tasks_.push_back(task);
        task_add_time_.push_back(task_add_time);
    }
}

template <typename TaskType>
std::shared_ptr<TaskType> Batch<TaskType>::RemoveTask()
{
    {
        std::lock_guard<bthread::Mutex> l(mu_);
        if (tasks_.empty()) {
            return nullptr;
        }
        std::shared_ptr<TaskType> task = tasks_.back();
        size_ -= task->size();
        tasks_.pop_back();
        return task;
    }
}

template <typename TaskType>
int Batch<TaskType>::num_tasks() const
{
    {
        std::lock_guard<bthread::Mutex> l(mu_);
        return tasks_.size();
    }
}

template <typename TaskType>
bool Batch<TaskType>::empty() const
{
    {
        std::lock_guard<bthread::Mutex> l(mu_);
        return tasks_.empty();
    }
}

template <typename TaskType>
const TaskType &Batch<TaskType>::task(int i) const
{
    DCHECK_GE(i, 0);
    {
        std::lock_guard<bthread::Mutex> l(mu_);
        DCHECK_LT(i, tasks_.size());
        return *tasks_[i].get();
    }
}

template <typename TaskType>
TaskType *Batch<TaskType>::mutable_task(int i)
{
    DCHECK_GE(i, 0);
    {
        std::lock_guard<bthread::Mutex> l(mu_);
        DCHECK_LT(i, tasks_.size());
        return tasks_[i].get();
    }
}

template <typename TaskType>
size_t Batch<TaskType>::size() const
{
    {
        std::lock_guard<bthread::Mutex> l(mu_);
        return size_;
    }
}

template <typename TaskType>
bool Batch<TaskType>::IsClosed() const
{
    return const_cast<absl::Notification *>(&closed_)->HasBeenNotified();
}

template <typename TaskType>
void Batch<TaskType>::WaitUntilClosed() const
{
    const_cast<absl::Notification *>(&closed_)->WaitForNotification();
}

template <typename TaskType>
void Batch<TaskType>::Close()
{
    closed_.Notify();
}

template <typename TaskType>
void Batch<TaskType>::SetTaskStartProcessTime(uint64_t task_start_process_time)
{
    task_start_process_time_ = task_start_process_time;
}

// template <typename TaskType>
// bthread_cond_t *Batch<TaskType>::GetBthreadCond()
// {
//     // std::cout << "GetBthreadCond, cond_: " << (void *)(&cond_) <<
//     std::endl; return &cond_;
// }

// template <typename TaskType>
// bthread_mutex_t *Batch<TaskType>::GetBthreadMutex()
// {
//     // std::cout << "GetBthreadMutex, mutex_: " << (void *)(&mutex_) <<
//     // std::endl;
//     return &mutex_;
// }

MERCURY_NAMESPACE_END(core);

#endif // MERCURY_CORE_FRAMEWORK_BATCH_SCHEDULER_H_