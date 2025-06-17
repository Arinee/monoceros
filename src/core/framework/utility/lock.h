/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     lock.h
 *   \author   Hechong.xyf
 *   \date     Jan 2018
 *   \version  1.0.0
 *   \brief    Interface of Mercury Utility Lock
 */

#ifndef __MERCURY_UTILITY_LOCK_H__
#define __MERCURY_UTILITY_LOCK_H__

#include <atomic>
#include <condition_variable>
#include <mutex>

namespace mercury {

/*! Spin Mutex
 */
class SpinMutex
{
public:
    //! Constructor
    SpinMutex(void) : _flag(false) {}

    //! Locking
    void lock(void)
    {
        while (_flag.test_and_set(std::memory_order_acquire))
            ;
    }

    //! Try locking
    bool try_lock(void)
    {
        return (!_flag.test_and_set(std::memory_order_acquire));
    }

    //! Unlocking
    void unlock(void)
    {
        _flag.clear(std::memory_order_release);
    }

private:
    //! Disable them
    SpinMutex(const SpinMutex &) = delete;
    SpinMutex(SpinMutex &&) = delete;
    SpinMutex &operator=(const SpinMutex &) = delete;
    SpinMutex &operator=(SpinMutex &&) = delete;

    //! Members
    std::atomic_flag _flag;
};

/*! Shared Mutex
 */
class SharedMutex
{
public:
    //! Constructor
    SharedMutex(void)
        : _pending_count(0), _read_count(0), _write_count(0), _mutex(),
          _read_cond(), _write_cond()
    {
    }

    //! Locking
    void lock(void)
    {
        std::unique_lock<std::mutex> q(_mutex);
        ++_write_count;
        _write_cond.wait(q, [this]() { return (_pending_count == 0); });
        --_write_count;
        --_pending_count;
    }

    //! Try locking
    bool try_lock(void)
    {
        std::unique_lock<std::mutex> q(_mutex, std::defer_lock);
        if (q.try_lock()) {
            if (_pending_count == 0) {
                --_pending_count;
                return true;
            }
        }
        return false;
    }

    //! Unlocking
    void unlock(void)
    {
        std::lock_guard<std::mutex> q(_mutex);
        ++_pending_count;

        if (_write_count != 0) {
            _write_cond.notify_one();
        } else {
            _read_cond.notify_all();
        }
    }

    //! Locking (shared)
    void lock_shared(void)
    {
        std::unique_lock<std::mutex> q(_mutex);
        ++_read_count;
        _read_cond.wait(
            q, [this]() { return (_write_count == 0 && _pending_count >= 0); });
        --_read_count;
        ++_pending_count;
    }

    //! Try locking (shared)
    bool try_lock_shared(void)
    {
        std::unique_lock<std::mutex> q(_mutex, std::defer_lock);
        if (q.try_lock()) {
            if (_write_count == 0 && _pending_count >= 0) {
                ++_pending_count;
                return true;
            }
        }
        return false;
    }

    //! Unlocking (shared)
    void unlock_shared(void)
    {
        std::lock_guard<std::mutex> q(_mutex);
        --_pending_count;

        if (_write_count != 0 && _pending_count == 0) {
            _write_cond.notify_one();
        } else {
            _read_cond.notify_all();
        }
    }

private:
    //! Disable them
    SharedMutex(const SharedMutex &) = delete;
    SharedMutex(SharedMutex &&) = delete;
    SharedMutex &operator=(const SharedMutex &) = delete;
    SharedMutex &operator=(SharedMutex &&) = delete;

    //! Members
    int32_t _pending_count;
    int32_t _read_count;
    int32_t _write_count;
    std::mutex _mutex;
    std::condition_variable _read_cond;
    std::condition_variable _write_cond;
};

/*! Write Lock
 */
class WriteLock
{
public:
    //! Constructor
    WriteLock(SharedMutex &mutex) : _mutex(mutex) {}

    //! Locking
    void lock(void)
    {
        _mutex.lock();
    }

    //! Try locking
    bool try_lock(void)
    {
        return _mutex.try_lock();
    }

    //! Unlocking
    void unlock(void)
    {
        _mutex.unlock();
    }

private:
    //! Disable them
    WriteLock(void) = delete;
    WriteLock(const WriteLock &) = delete;
    WriteLock(WriteLock &&) = delete;
    WriteLock &operator=(const WriteLock &) = delete;
    WriteLock &operator=(WriteLock &&) = delete;

    //! Members
    SharedMutex &_mutex;
};

/*! Read Lock
 */
class ReadLock
{
public:
    //! Constructor
    ReadLock(SharedMutex &mutex) : _mutex(mutex) {}

    //! Locking
    void lock(void)
    {
        _mutex.lock_shared();
    }

    //! Try locking
    bool try_lock(void)
    {
        return _mutex.try_lock_shared();
    }

    //! Unlocking
    void unlock(void)
    {
        _mutex.unlock_shared();
    }

private:
    //! Disable them
    ReadLock(void) = delete;
    ReadLock(const ReadLock &) = delete;
    ReadLock(ReadLock &&) = delete;
    ReadLock &operator=(const ReadLock &) = delete;
    ReadLock &operator=(ReadLock &&) = delete;

    //! Members
    SharedMutex &_mutex;
};

} // namespace mercury

#endif // __MERCURY_UTILITY_LOCK_H__
