/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     time_helper.h
 *   \author   Hechong.xyf
 *   \date     Dec 2017
 *   \version  1.0.0
 *   \brief    Interface of Mercury Utility Time Helper
 */

#ifndef __MERCURY_UTILITY_TIME_HELPER_H__
#define __MERCURY_UTILITY_TIME_HELPER_H__

#include "internal/platform.h"

namespace mercury {

/*! Monotime
 */
struct Monotime
{
    //! Retrieve monotonic time in nanoseconds
    static uint64_t NanoSeconds(void);

    //! Retrieve monotonic time in microseconds
    static uint64_t MicroSeconds(void);

    //! Retrieve monotonic time in milliseconds
    static uint64_t MilliSeconds(void);

    //! Retrieve monotonic time in seconds
    static uint64_t Seconds(void);
};

/*! Realtime
 */
struct Realtime
{
    //! Retrieve system time in nanoseconds
    static uint64_t NanoSeconds(void);

    //! Retrieve system time in microseconds
    static uint64_t MicroSeconds(void);

    //! Retrieve system time in milliseconds
    static uint64_t MilliSeconds(void);

    //! Retrieve system time in seconds
    static uint64_t Seconds(void);
};

/*! Elapsed Time
 */
class ElapsedTime
{
public:
    //! Constructor
    ElapsedTime(void) : _stamp(Monotime::MilliSeconds()) {}

    //! Update time stamp
    size_t update(void)
    {
        uint64_t old_stamp = _stamp;
        _stamp = Monotime::MilliSeconds();
        return (size_t)(_stamp - old_stamp);
    }

    //! Retrieve the elapsed time in milliseconds
    size_t elapsed(void) const
    {
        return (size_t)(Monotime::MilliSeconds() - _stamp);
    }

private:
    uint64_t _stamp;
};

} // namespace mercury

#endif // __MERCURY_UTILITY_TIME_HELPER_H__
