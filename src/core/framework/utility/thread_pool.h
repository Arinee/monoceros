/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     thread_pool.h
 *   \author   Hechong.xyf
 *   \date     Jan 2018
 *   \version  1.0.0
 *   \brief    Interface of Mercury Utility Thread Pool
 */

#ifndef __MERCURY_UTILITY_THREAD_POOL_H__
#define __MERCURY_UTILITY_THREAD_POOL_H__

#if defined(__linux) || defined(__linux__)
#include <pthread.h>
#endif

#include "closure.h"
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace mercury {

/*! Thread Pool
 */
class ThreadPool
{
public:
    //! Constructor
    ThreadPool(void)
        : ThreadPool{ true, std::max(std::thread::hardware_concurrency(), 1u) }
    {
    }

    //! Constructor
    explicit ThreadPool(bool binding)
        : ThreadPool{ binding,
                      std::max(std::thread::hardware_concurrency(), 1u) }
    {
    }

    //! Constructor
    explicit ThreadPool(bool binding, uint32_t size)
        : _queue(), _stopping(false), _active_count(0), _pending_count(0),
          _queue_mutex(), _finish_mutex(), _work_cond(), _finished_cond(),
          _pool()
    {
        for (uint32_t i = 0u; i < size; ++i) {
            _pool.emplace_back(&ThreadPool::worker, this);
        }
        if (binding) {
            this->bind();
        }
    }

    //! Destructor
    ~ThreadPool(void)
    {
        this->stop();

        // Join all threads
        for (auto it = _pool.begin(); it != _pool.end(); ++it) {
            if (it->joinable()) {
                it->join();
            }
        }
    }

    //! Retrieve thread count in pool
    size_t count(void) const
    {
        return _pool.size();
    }

    //! Stop all threads
    void stop(void)
    {
        // Set stop flag as ture, then wake all threads
        _stopping = true;
        std::lock_guard<std::mutex> lock(_queue_mutex);
        _work_cond.notify_all();
    }

    //! Push a task to the queue
    void enqueue(const Closure::Pointer &task, bool wake)
    {
        std::lock_guard<std::mutex> lock(_queue_mutex);
        ++_pending_count;
        _queue.push(task);
        if (wake) {
            _work_cond.notify_one();
        }
    }

    //! Push a task to the queue
    void enqueue(Closure::Pointer &&task, bool wake)
    {
        std::lock_guard<std::mutex> lock(_queue_mutex);
        ++_pending_count;
        _queue.push(std::move(task));
        if (wake) {
            _work_cond.notify_one();
        }
    }

    //! Push a task to the queue
    void enqueue(const Closure::Pointer &task)
    {
        this->enqueue(task, false);
    }

    //! Push a task to the queue
    void enqueue(Closure::Pointer &&task)
    {
        this->enqueue(std::move(task), false);
    }

    //! Wake any one thread
    void wakeAny(void)
    {
        std::lock_guard<std::mutex> lock(_queue_mutex);
        _work_cond.notify_one();
    }

    //! Wake all threads
    void wakeAll(void)
    {
        std::lock_guard<std::mutex> lock(_queue_mutex);
        _work_cond.notify_all();
    }

    //! Wait until all threads finished processing
    void waitFinish(void)
    {
        std::unique_lock<std::mutex> lock(_finish_mutex);
        _finished_cond.wait(lock, [this]() { return this->isFinished(); });
    }

    //! Check if the pool is finished
    bool isFinished(void) const
    {
        return (_active_count == 0 && _pending_count == 0);
    }

    //! Retrieve count of pending tasks in pool
    size_t getPendingCount(void) const
    {
        return _pending_count.load(std::memory_order_relaxed);
    }

    //! Retrieve count of active tasks in pool
    size_t getActiveCount(void) const
    {
        return _active_count.load(std::memory_order_relaxed);
    }

    //! Get the thread index via thread id
    int getIndex(const std::thread::id &thread_id) const
    {
        for (size_t i = 0; i < _pool.size(); ++i) {
            if (_pool[i].get_id() == thread_id) {
                return static_cast<int>(i);
            }
        }
        return -1;
    }

    //! Get the current work thread index
    int getThisIndex(void) const
    {
        return this->getIndex(std::this_thread::get_id());
    }

protected:
    //! Thread worker callback
    void worker(void)
    {
        for (;;) {
            Closure::Pointer task;

            if (!this->picking(&task)) {
                break;
            }

            // Run the task
            if (task) {
                task->run();
            }

            // Decrease count of active works
            std::lock_guard<std::mutex> lock(_finish_mutex);
            if (--_active_count == 0 && _pending_count == 0) {
                _finished_cond.notify_all();
            }
        }
    }

    //! Pick a task from queue
    bool picking(Closure::Pointer *task)
    {
        std::unique_lock<std::mutex> latch(_queue_mutex);
        _work_cond.wait(latch,
                        [this]() { return (_pending_count > 0 || _stopping); });

        if (_stopping) {
            return false;
        }

        // Counter of active tasks
        ++_active_count;

        // Pop a task
        *task = std::move(_queue.front());
        _queue.pop();
        --_pending_count;
        return true;
    }

    //! Bind threads to processors
    void bind(void)
    {
#if defined(__linux) || defined(__linux__)
        uint32_t hc = std::thread::hardware_concurrency();
        if (hc > 1) {
            cpu_set_t mask;
            for (uint32_t i = 0u; i < _pool.size(); ++i) {
                CPU_ZERO(&mask);
                CPU_SET(i % hc, &mask);
                pthread_setaffinity_np(_pool[i].native_handle(), sizeof(mask),
                                       &mask);
            }
        }
#endif
    }

private:
    //! Disable them
    ThreadPool(const ThreadPool &) = delete;
    ThreadPool &operator=(const ThreadPool &) = delete;

    //! Members
    std::queue<Closure::Pointer> _queue;
    std::atomic_bool _stopping;
    std::atomic_uint _active_count;
    std::atomic_uint _pending_count;
    std::mutex _queue_mutex;
    std::mutex _finish_mutex;
    std::condition_variable _work_cond;
    std::condition_variable _finished_cond;
    std::vector<std::thread> _pool;
};

} // namespace mercury

#endif // __MERCURY_UTILITY_THREAD_POOL_H__