/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     version.h
 *   \author   Hechong.xyf
 *   \date     Apr 2018
 *   \version  1.0.0
 *   \brief    Interface of Mercury Version
 */

#ifndef __MERCURY_VERSION_H__
#define __MERCURY_VERSION_H__

#define MERCURY_VERSION_MAJOR 1
#define MERCURY_VERSION_MINOR 0
#define MERCURY_VERSION_MICRO 9
#define MERCURY_VERSION_STRING "1.0.9"

#define MERCURY_VERSION_LEVEL_ALPHA 0xA
#define MERCURY_VERSION_LEVEL_BETA 0xB
#define MERCURY_VERSION_LEVEL_GAMMA 0xC /* For release candidates */
#define MERCURY_VERSION_LEVEL_FINAL 0xF /* Serial should be 0 here */
#define MERCURY_VERSION_LEVEL MERCURY_VERSION_LEVEL_FINAL
#define MERCURY_VERSION_SERIAL 0

/*! Version as a single 4-byte hex number, e.g. 0x010502B2 == 1.5.2b2.
 *  Use this for numeric comparisons, e.g. #if PY_VERSION_HEX >= ...
 */
#define MERCURY_VERSION_HEX                                                    \
    ((MERCURY_VERSION_MAJOR << 24) | (MERCURY_VERSION_MINOR << 16) |           \
     (MERCURY_VERSION_MICRO << 8) | (MERCURY_VERSION_LEVEL << 4) |             \
     (MERCURY_VERSION_SERIAL << 0))


#ifndef MERCURY_STRING_
#define MERCURY_STRING_(x) #x
#endif

#ifndef MERCURY_STRING
#define MERCURY_STRING(x) MERCURY_STRING_(x)
#endif

/*! http://nadeausoftware.com/articles/2012/01/
 *  c_c_tip_how_use_compiler_predefined_macros_detect_operating_system
 */
#if defined(__linux) || defined(__linux__)
#define MERCURY_VERSION_PLATFORM "Linux"
#elif defined(__FreeBSD__)
#define MERCURY_VERSION_PLATFORM "FreeBSD"
#elif defined(__NetBSD__)
#define MERCURY_VERSION_PLATFORM "NetBSD"
#elif defined(__OpenBSD__)
#define MERCURY_VERSION_PLATFORM "OpenBSD"
#elif defined(__APPLE__) || defined(__MACH__)
#define MERCURY_VERSION_PLATFORM "Darwin"
#elif defined(__CYGWIN__) && !defined(_WIN32)
#define MERCURY_VERSION_PLATFORM "Cygwin"
#elif defined(_WIN64)
#define MERCURY_VERSION_PLATFORM "Microsoft Windows (64-bit)"
#elif defined(_WIN32)
#define MERCURY_VERSION_PLATFORM "Microsoft Windows (32-bit)"
#elif defined(__sun) && defined(__SVR4)
#define MERCURY_VERSION_PLATFORM "Solaris"
#elif defined(_AIX)
#define MERCURY_VERSION_PLATFORM "AIX"
#elif defined(__hpux)
#define MERCURY_VERSION_PLATFORM "HP-UX"
#elif defined(__unix) || defined(__unix__)
#define MERCURY_VERSION_PLATFORM "Unix"
#else
#define MERCURY_VERSION_PLATFORM "Unknown Platform"
#endif

/*! http://nadeausoftware.com/articles/2012/10/
 *  c_c_tip_how_detect_compiler_name_and_version_using_compiler_predefined_macros
 */
#if defined(__clang__)
#define MERCURY_VERSION_COMPILER_NAME "Clang/LLVM"
#elif defined(__ICC) || defined(__INTEL_COMPILER)
#define MERCURY_VERSION_COMPILER_NAME "Intel ICC/ICPC"
#elif defined(__GNUC__) || defined(__GNUG__)
#define MERCURY_VERSION_COMPILER_NAME "GNU GCC/G++"
#elif defined(__HP_cc) || defined(__HP_aCC)
#define MERCURY_VERSION_COMPILER_NAME "Hewlett-Packard C/aC++"
#elif defined(__IBMC__) || defined(__IBMCPP__)
#define MERCURY_VERSION_COMPILER_NAME "IBM XL C/C++"
#elif defined(_MSC_VER)
#define MERCURY_VERSION_COMPILER_NAME "Microsoft Visual C++"
#elif defined(__PGI)
#define MERCURY_VERSION_COMPILER_NAME "Portland Group PGCC/PGCPP"
#elif defined(__SUNPRO_C) || defined(__SUNPRO_CC)
#define MERCURY_VERSION_COMPILER_NAME "Oracle Solaris Studio"
#else
#define MERCURY_VERSION_COMPILER_NAME "Unknown Compiler"
#endif

#if defined(__VERSION__)
#define MERCURY_VERSION_COMPILER                                               \
    MERCURY_VERSION_COMPILER_NAME " (" __VERSION__ ")"
#elif defined(_MSC_FULL_VER)
#define MERCURY_VERSION_COMPILER                                               \
    MERCURY_VERSION_COMPILER_NAME " (" MERCURY_STRING(_MSC_FULL_VER) ")"
#elif defined(_MSC_VER)
#define MERCURY_VERSION_COMPILER                                               \
    MERCURY_VERSION_COMPILER_NAME " (" MERCURY_STRING(_MSC_VER) ")"
#elif defined(__PGIC__)
#define MERCURY_VERSION_COMPILER                                               \
    MERCURY_VERSION_COMPILER_NAME                                              \
    " (" MERCURY_STRING(__PGIC__) "." MERCURY_STRING(                          \
        __PGIC_MINOR__) "." MERCURY_STRING(__PGIC_PATCHLEVEL__) ")"
#elif defined(__xlc__)
#define MERCURY_VERSION_COMPILER MERCURY_VERSION_COMPILER_NAME " (" __xlc__ ")"
#elif defined(__SUNPRO_C)
#define MERCURY_VERSION_COMPILER                                               \
    MERCURY_VERSION_COMPILER_NAME " (" MERCURY_STRING(__SUNPRO_C) ")"
#elif defined(__HP_cc)
#define MERCURY_VERSION_COMPILER                                               \
    MERCURY_VERSION_COMPILER_NAME " (" MERCURY_STRING(__HP_cc) ")"
#else
#define MERCURY_VERSION_COMPILER MERCURY_VERSION_COMPILER_NAME
#endif

#if defined(__x86_64__) || defined(_M_X64)
#define MERCURY_VERSION_PROCESSOR "x86 64-bit Processor"
#elif defined(__i386) || defined(_M_IX86)
#define MERCURY_VERSION_PROCESSOR "x86 32-bit Processor"
#elif defined(__ia64) || defined(__itanium__) || defined(_M_IA64)
#define MERCURY_VERSION_PROCESSOR "Itanium Processor"
#elif defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)
#define MERCURY_VERSION_PROCESSOR "PowerPC 64-bit Processor"
#elif defined(__powerpc__) || defined(__ppc__) || defined(__PPC__)
#define MERCURY_VERSION_PROCESSOR "PowerPC 32-bit Processor"
#elif defined(__sparc)
#define MERCURY_VERSION_PROCESSOR "SPARC Processor"
#else
#define MERCURY_VERSION_PROCESSOR "Unknown Processor"
#endif

namespace mercury {

/*! Mercury Version
 */
struct Version
{
    //! Retrieve the version number in string
    static const char *String(void);

    //! Retrieve the version number in HEX
    static unsigned int Hex(void);

    //! Retrieve the detailed version information
    static const char *Details(void);
};

} // namespace mercury

#endif // __MERCURY_VERSION_H__
