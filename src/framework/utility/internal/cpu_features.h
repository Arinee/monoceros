/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     cpu_features.h
 *   \author   Hechong.xyf
 *   \date     Nov 2017
 *   \version  1.0.0
 *   \brief    Interface of Mercury CPU Features
 */

#ifndef __MERCURY_UTILITY_INTERNAL_CPU_FEATURES_H__
#define __MERCURY_UTILITY_INTERNAL_CPU_FEATURES_H__

#include "platform.h"

namespace mercury {
namespace internal {

/*! Cpu Features
 */
class CpuFeatures
{
public:
    //! Multimedia Extensions
    static bool MMX(void)
    {
        return !!(_flags.l1_edx & (1u << 23));
    }

    //! Streaming SIMD Extensions
    static bool SSE(void)
    {
        return !!(_flags.l1_edx & (1u << 25));
    }

    //! Streaming SIMD Extensions 2
    static bool SSE2(void)
    {
        return !!(_flags.l1_edx & (1u << 26));
    }

    //! Streaming SIMD Extensions 3
    static bool SSE3(void)
    {
        return !!(_flags.l1_ecx & (1u << 0));
    }

    //! Supplemental Streaming SIMD Extensions 3
    static bool SSSE3(void)
    {
        return !!(_flags.l1_ecx & (1u << 9));
    }

    //! Streaming SIMD Extensions 4.1
    static bool SSE4_1(void)
    {
        return !!(_flags.l1_ecx & (1u << 19));
    }

    //! Streaming SIMD Extensions 4.2
    static bool SSE4_2(void)
    {
        return !!(_flags.l1_ecx & (1u << 20));
    }

    //! Advanced Vector Extensions
    static bool AVX(void)
    {
        return !!(_flags.l1_ecx & (1u << 28));
    }

    //! Advanced Vector Extensions 2
    static bool AVX2(void)
    {
        return !!(_flags.l7_ebx & (1u << 5));
    }

    //! AVX-512 Foundation
    static bool AVX512F(void)
    {
        return !!(_flags.l7_ebx & (1u << 16));
    }

    //! AVX-512 DQ (Double/Quad granular) Instructions
    static bool AVX512DQ(void)
    {
        return !!(_flags.l7_ebx & (1u << 17));
    }

    //! AVX-512 Prefetch
    static bool AVX512PF(void)
    {
        return !!(_flags.l7_ebx & (1u << 26));
    }

    //! AVX-512 Exponential and Reciprocal
    static bool AVX512ER(void)
    {
        return !!(_flags.l7_ebx & (1u << 27));
    }

    //! AVX-512 Conflict Detection
    static bool AVX512CD(void)
    {
        return !!(_flags.l7_ebx & (1u << 28));
    }

    //! AVX-512 BW (Byte/Word granular) Instructions
    static bool AVX512BW(void)
    {
        return !!(_flags.l7_ebx & (1u << 30));
    }

    //! AVX-512 VL (128/256 Vector Length) Extensions
    static bool AVX512VL(void)
    {
        return !!(_flags.l7_ebx & (1u << 31));
    }

    //! AVX-512 Integer Fused Multiply-Add instructions
    static bool AVX512_IFMA(void)
    {
        return !!(_flags.l7_ebx & (1u << 21));
    }

    //! AVX512 Vector Bit Manipulation instructions
    static bool AVX512_VBMI(void)
    {
        return !!(_flags.l7_ecx & (1u << 1));
    }

    //! Additional AVX512 Vector Bit Manipulation Instructions
    static bool AVX512_VBMI2(void)
    {
        return !!(_flags.l7_ecx & (1u << 6));
    }

    //! Vector Neural Network Instructions
    static bool AVX512_VNNI(void)
    {
        return !!(_flags.l7_ecx & (1u << 11));
    }

    //! Support for VPOPCNT[B,W] and VPSHUF-BITQMB instructions
    static bool AVX512_BITALG(void)
    {
        return !!(_flags.l7_ecx & (1u << 12));
    }

    //! POPCNT for vectors of DW/QW
    static bool AVX512_VPOPCNTDQ(void)
    {
        return !!(_flags.l7_ecx & (1u << 14));
    }

    //! AVX-512 Neural Network Instructions
    static bool AVX512_4VNNIW(void)
    {
        return !!(_flags.l7_edx & (1u << 2));
    }

    //! AVX-512 Multiply Accumulation Single precision
    static bool AVX512_4FMAPS(void)
    {
        return !!(_flags.l7_edx & (1u << 3));
    }

    //! CMPXCHG8 instruction
    static bool CX8(void)
    {
        return !!(_flags.l1_edx & (1u << 8));
    }

    //! CMPXCHG16B instruction
    static bool CX16(void)
    {
        return !!(_flags.l1_ecx & (1u << 13));
    }

    //! PCLMULQDQ instruction
    static bool PCLMULQDQ(void)
    {
        return !!(_flags.l1_ecx & (1u << 1));
    }

    //! Carry-Less Multiplication Double Quadword
    static bool VPCLMULQDQ(void)
    {
        return !!(_flags.l7_ecx & (1u << 10));
    }

    //! CMOV instructions (plus FCMOVcc, FCOMI with FPU)
    static bool CMOV(void)
    {
        return !!(_flags.l1_edx & (1u << 15));
    }

    //! MOVBE instruction
    static bool MOVBE(void)
    {
        return !!(_flags.l1_ecx & (1u << 22));
    }

    //! Enhanced REP MOVSB/STOSB instructions
    static bool ERMS(void)
    {
        return !!(_flags.l7_ebx & (1u << 9));
    }

    //! POPCNT instruction
    static bool POPCNT(void)
    {
        return !!(_flags.l1_ecx & (1u << 23));
    }

    //! XSAVE/XRSTOR/XSETBV/XGETBV instructions
    static bool XSAVE(void)
    {
        return !!(_flags.l1_ecx & (1u << 26));
    }

    //! Fused multiply-add
    static bool FMA(void)
    {
        return !!(_flags.l1_ecx & (1u << 12));
    }

    //! ADCX and ADOX instructions
    static bool ADX(void)
    {
        return !!(_flags.l7_ebx & (1u << 19));
    }

    //! Galois Field New Instructions
    static bool GFNI(void)
    {
        return !!(_flags.l7_ecx & (1u << 8));
    }

    //! AES instructions
    static bool AES(void)
    {
        return !!(_flags.l1_ecx & (1u << 25));
    }

    //! Vector AES
    static bool VAES(void)
    {
        return !!(_flags.l7_ecx & (1u << 9));
    }

    //! RDSEED instruction
    static bool RDSEED(void)
    {
        return !!(_flags.l7_ebx & (1u << 18));
    }

    //! RDRAND instruction
    static bool RDRAND(void)
    {
        return !!(_flags.l1_ecx & (1u << 30));
    }

    //! SHA1/SHA256 Instruction Extensions
    static bool SHA(void)
    {
        return !!(_flags.l7_ebx & (1u << 29));
    }

    //! 1st group bit manipulation extensions
    static bool BMI1(void)
    {
        return !!(_flags.l7_ebx & (1u << 3));
    }

    //! 2nd group bit manipulation extensions
    static bool BMI2(void)
    {
        return !!(_flags.l7_ebx & (1u << 8));
    }

    //! CLFLUSH instruction
    static bool CLFLUSH(void)
    {
        return !!(_flags.l1_edx & (1u << 19));
    }

    //! CLFLUSHOPT instruction
    static bool CLFLUSHOPT(void)
    {
        return !!(_flags.l7_ebx & (1u << 23));
    }

    //! CLWB instruction
    static bool CLWB(void)
    {
        return !!(_flags.l7_ebx & (1u << 24));
    }

    //! RDPID instruction
    static bool RDPID(void)
    {
        return !!(_flags.l7_ecx & (1u << 22));
    }

    //! Onboard FPU
    static bool FPU(void)
    {
        return !!(_flags.l1_edx & (1u << 0));
    }

    //! Hyper-Threading
    static bool HT(void)
    {
        return !!(_flags.l1_edx & (1u << 28));
    }

    //! Hardware virtualization
    static bool VMX(void)
    {
        return !!(_flags.l1_ecx & (1u << 5));
    }

    //！Running on a hypervisor
    static bool HYPERVISOR(void)
    {
        return !!(_flags.l1_ecx & (1u << 31));
    }

private:
    struct CpuFlags
    {
        //! Constructor
        CpuFlags(void);

        //! Members
        uint32_t l1_ecx;
        uint32_t l1_edx;
        uint32_t l7_ebx;
        uint32_t l7_ecx;
        uint32_t l7_edx;
    };

    //! Static Members
    static CpuFlags _flags;
};

} // namespace internal
} // namespace mercury

#endif // __MERCURY_UTILITY_INTERNAL_CPU_FEATURES_H__
