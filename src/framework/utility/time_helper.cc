/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     time_helper.h
 *   \author   Hechong.xyf
 *   \date     Dec 2017
 *   \version  1.0.0
 *   \brief    Implementation of Mercury Utility Time Helper
 */

#include "time_helper.h"

#if defined(_WIN64) || defined(_WIN32)
#include <Windows.h>
#endif

namespace mercury {

#if defined(_WIN64) || defined(_WIN32)
uint64_t Monotime::NanoSeconds(void)
{
    LARGE_INTEGER stamp, freq;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&stamp);
    return (uint64_t)((double)stamp.QuadPart *
                      (1000000000.0 / (double)freq.QuadPart));
}

uint64_t Monotime::MicroSeconds(void)
{
    LARGE_INTEGER stamp, freq;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&stamp);
    return (stamp.QuadPart * 1000000u / freq.QuadPart);
}

uint64_t Monotime::MilliSeconds(void)
{
    LARGE_INTEGER stamp, freq;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&stamp);
    return (stamp.QuadPart * 1000u / freq.QuadPart);
}

uint64_t Monotime::Seconds(void)
{
    LARGE_INTEGER stamp, freq;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&stamp);
    return (stamp.QuadPart / freq.QuadPart);
}

// January 1, 1970 (start of Unix epoch) in "ticks"
#define UNIX_TIME_START 0x019DB1DED53E8000ull

uint64_t Realtime::NanoSeconds(void)
{
    LARGE_INTEGER stamp;
    FILETIME file;
    GetSystemTimeAsFileTime(&file);
    stamp.HighPart = file.dwHighDateTime;
    stamp.LowPart = file.dwLowDateTime;
    return (stamp.QuadPart - UNIX_TIME_START) * 100u;
}

uint64_t Realtime::MicroSeconds(void)
{
    LARGE_INTEGER stamp;
    FILETIME file;
    GetSystemTimeAsFileTime(&file);
    stamp.HighPart = file.dwHighDateTime;
    stamp.LowPart = file.dwLowDateTime;
    return (stamp.QuadPart - UNIX_TIME_START) / 10u;
}

uint64_t Realtime::MilliSeconds(void)
{
    LARGE_INTEGER stamp;
    FILETIME file;
    GetSystemTimeAsFileTime(&file);
    stamp.HighPart = file.dwHighDateTime;
    stamp.LowPart = file.dwLowDateTime;
    return (stamp.QuadPart - UNIX_TIME_START) / 10000u;
}

uint64_t Realtime::Seconds(void)
{
    LARGE_INTEGER stamp;
    FILETIME file;
    GetSystemTimeAsFileTime(&file);
    stamp.HighPart = file.dwHighDateTime;
    stamp.LowPart = file.dwLowDateTime;
    return (stamp.QuadPart - UNIX_TIME_START) / 10000000u;
}
#else
uint64_t Monotime::NanoSeconds(void)
{
    struct timespec tspec;
    clock_gettime(CLOCK_MONOTONIC, &tspec);
    return (tspec.tv_sec * 1000000000u + tspec.tv_nsec);
}

uint64_t Monotime::MicroSeconds(void)
{
    struct timespec tspec;
    clock_gettime(CLOCK_MONOTONIC, &tspec);
    return (tspec.tv_sec * 1000000u + tspec.tv_nsec / 1000u);
}

uint64_t Monotime::MilliSeconds(void)
{
    struct timespec tspec;
    clock_gettime(CLOCK_MONOTONIC, &tspec);
    return (tspec.tv_sec * 1000u + tspec.tv_nsec / 1000000u);
}

uint64_t Monotime::Seconds(void)
{
    struct timespec tspec;
    clock_gettime(CLOCK_MONOTONIC, &tspec);
    return (tspec.tv_sec);
}

uint64_t Realtime::NanoSeconds(void)
{
    struct timespec tspec;
    clock_gettime(CLOCK_REALTIME, &tspec);
    return (tspec.tv_sec * 1000000000u + tspec.tv_nsec);
}

uint64_t Realtime::MicroSeconds(void)
{
    struct timespec tspec;
    clock_gettime(CLOCK_REALTIME, &tspec);
    return (tspec.tv_sec * 1000000u + tspec.tv_nsec / 1000u);
}

uint64_t Realtime::MilliSeconds(void)
{
    struct timespec tspec;
    clock_gettime(CLOCK_REALTIME, &tspec);
    return (tspec.tv_sec * 1000u + tspec.tv_nsec / 1000000u);
}

uint64_t Realtime::Seconds(void)
{
    struct timespec tspec;
    clock_gettime(CLOCK_REALTIME, &tspec);
    return (tspec.tv_sec);
}
#endif // _WIN64 || _WIN32

} // namespace mercury
