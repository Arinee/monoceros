/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     version.cc
 *   \author   Hechong.xyf
 *   \date     Apr 2018
 *   \version  1.0.0
 *   \brief    Implementation of Mercury Version
 */

#include "version.h"
#include "utility/internal/platform.h"

namespace mercury {
    
static const char MERCURY_VERSION_DETAILS[] =
    "Mercury Library " MERCURY_VERSION_STRING ".\n"
    "Copyright (C) The Software Authors. All rights reserved.\n"
    "Compiled by " MERCURY_VERSION_COMPILER ".\n"
    "Compiled for " MERCURY_VERSION_PROCESSOR ".\n"
    "Compiled on " MERCURY_VERSION_PLATFORM " on " __DATE__ " " __TIME__ ".\n"
    "Compiled with: \n"

#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    "    Little-endian Byte Order\n"
#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    "    Big-endian Byte Order\n"
#elif __BYTE_ORDER__ == __ORDER_PDP_ENDIAN__
    "    PDP-endian Byte Order\n"
#endif

#if defined(_DEBUG) || (!defined(__OPTIMIZE__) && !defined(NDEBUG))
    "    Debug Information\n"
#endif

#if defined(__STDC_VERSION__)
    "    C Standard " MERCURY_STRING(__STDC_VERSION__) "\n"
#endif

#if defined(__cplusplus)
    "    C++ Standard " MERCURY_STRING(__cplusplus) "\n"
#endif

#if defined(__GLIBC__)
    "    GNU glibc " MERCURY_STRING(__GLIBC__) "." 
    MERCURY_STRING(__GLIBC_MINOR__) "\n"
#endif

#if defined(WINVER)
    "    Microsoft Windows SDK " MERCURY_STRING(WINVER) "\n"
#endif

#if defined(__CLR_VER)
    "    Microsoft CLR " MERCURY_STRING(__CLR_VER) "\n"
#endif

#if defined(_POSIX_VERSION)
    "    POSIX " MERCURY_STRING(_POSIX_VERSION) "\n"
#endif

#if defined(__FMA__)
    "    Intel Intrinsics FMA\n"
#endif

#if defined(__AVX512F__)
    "    Intel Intrinsics AVX512F\n"
#elif defined(__AVX2__)
    "    Intel Intrinsics AVX2\n"
#elif defined(__AVX__)
    "    Intel Intrinsics AVX\n"
#elif defined(__SSE4_2__)
    "    Intel Intrinsics SSE4.2\n"
#elif defined(__SSE4_1__)
    "    Intel Intrinsics SSE4.1\n"
#elif defined(__SSSE3__)
    "    Intel Intrinsics SSSE3\n"
#elif defined(__SSE3__)
    "    Intel Intrinsics SSE3\n"
#elif defined(__SSE2__)
    "    Intel Intrinsics SSE2\n"
#elif defined(__SSE__)
    "    Intel Intrinsics SSE\n"
#endif

#if defined(_OPENMP)
    "    OpenMP API " MERCURY_STRING(_OPENMP) "\n"
#endif
    "\n";

const char *Version::String(void)
{
    return MERCURY_VERSION_STRING;
}

unsigned int Version::Hex(void)
{
    return MERCURY_VERSION_HEX;
}

const char *Version::Details(void)
{
    return MERCURY_VERSION_DETAILS;
}

} // namespace mercury
