/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     cpu_features.cc
 *   \author   Hechong.xyf
 *   \date     Nov 2017
 *   \version  1.0.0
 *   \brief    Implementation of Mercury CPU Features
 */

#include "cpu_features.h"

#if !defined(_MSC_VER)
#include <cpuid.h>
#endif

namespace mercury {
namespace internal {

//
// REFER: https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/
//        tree/arch/x86/include/asm/cpufeatures.h
//        https://software.intel.com/sites/default/files/managed/c5/15/
//        architecture-instruction-set-extensions-programming-reference.pdf
//

CpuFeatures::CpuFlags CpuFeatures::_flags;

#if defined(_MSC_VER)
CpuFeatures::CpuFlags::CpuFlags(void)
    : l1_ecx(0), l1_edx(0), l7_ebx(0), l7_ecx(0), l7_edx(0)
{
    int l1[4] = { 0, 0, 0, 0 };
    int l7[4] = { 0, 0, 0, 0 };

    __cpuidex(l1, 1, 0);
    __cpuidex(l7, 7, 0);
    l1_ecx = l1[2];
    l1_edx = l1[3];
    l7_ebx = l7[1];
    l7_ecx = l7[2];
    l7_edx = l7[3];
}
#else
CpuFeatures::CpuFlags::CpuFlags(void)
    : l1_ecx(0), l1_edx(0), l7_ebx(0), l7_ecx(0), l7_edx(0)
{
    uint32_t eax, ebx, ecx, edx;

    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        l1_ecx = ecx;
        l1_edx = edx;
    }
    if (__get_cpuid_max(0, NULL) >= 7) {
        __cpuid_count(7, 0, eax, ebx, ecx, edx);
        l7_ebx = ebx;
        l7_ecx = ecx;
        l7_edx = edx;
    }
}
#endif

} // namespace internal
} // namespace mercury
