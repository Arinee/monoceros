/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     fixed_bitset.h
 *   \author   Hechong.xyf
 *   \date     Mar 2018
 *   \version  1.0.0
 *   \brief    Interface of Mercury Utility Fixed Bitset
 */

#ifndef __MERCURY_UTILITY_FIXED_BITSET_H__
#define __MERCURY_UTILITY_FIXED_BITSET_H__

#include "internal/bitset.h"

namespace mercury {

/*! Fixed Bitset Module
 */
template <size_t N, typename = typename std::enable_if<N % 32 == 0>::type>
class FixedBitset
{
public:
    enum
    {
        MAX_SIZE = N
    };

    //! Constructor
    FixedBitset(void)
    {
        memset(_arr, 0, sizeof(_arr));
    }

    //! Constructor
    FixedBitset(const FixedBitset &rhs)
    {
        memcpy(_arr, rhs._arr, sizeof(_arr));
    }

    //! Destructor
    ~FixedBitset(void) {}

    //! Assignment
    FixedBitset &operator=(const FixedBitset &rhs)
    {
        memcpy(_arr, rhs._arr, sizeof(_arr));
        return *this;
    }

    //! Retrieve data pointer
    uint32_t *data(void)
    {
        return reinterpret_cast<uint32_t *>(_arr);
    }

    //! Retrieve data pointer
    const uint32_t *data(void) const
    {
        return reinterpret_cast<const uint32_t *>(_arr);
    }

    //! Retrieve count of bits in set
    constexpr size_t size(void) const
    {
        return MAX_SIZE;
    }

    //! Copy the content from another bitset
    void copy(const FixedBitset &rhs)
    {
        memcpy(_arr, rhs._arr, sizeof(_arr));
    }

    //! Test a bit in bitset
    bool test(size_t num) const
    {
        platform_assert_with(N > num, "overflow argument");
        return ((_arr[num >> 5] & (1u << (num & 0x1f))) != 0);
    }

    //! Set a bit in bitset
    void set(size_t num)
    {
        platform_assert_with(N > num, "overflow argument");
        uint32_t mask = (1u << (num & 0x1f));
        _arr[num >> 5] |= mask;
    }

    //! Clear a bit in bitset
    void reset(size_t num)
    {
        platform_assert_with(N > num, "overflow argument");
        uint32_t mask = (1u << (num & 0x1f));
        _arr[num >> 5] &= ~mask;
    }

    //! Toggle a bit in bitset
    void flip(size_t num)
    {
        platform_assert_with(N > num, "overflow argument");
        uint32_t mask = (1u << (num & 0x1f));
        _arr[num >> 5] ^= mask;
    }

    //! Perform binary AND
    void performAnd(const FixedBitset &rhs)
    {
        mercury::internal::BitsetAnd(_arr, rhs._arr, ((N + 0x1f) >> 5));
    }

    //! Perform binary AND NOT
    void performAndnot(const FixedBitset &rhs)
    {
        mercury::internal::BitsetAndnot(_arr, rhs._arr, ((N + 0x1f) >> 5));
    }

    //! Perform binary OR
    void performOr(const FixedBitset &rhs)
    {
        mercury::internal::BitsetOr(_arr, rhs._arr, ((N + 0x1f) >> 5));
    }

    //! Perform binary XOR
    void performXor(const FixedBitset &rhs)
    {
        mercury::internal::BitsetXor(_arr, rhs._arr, ((N + 0x1f) >> 5));
    }

    //! Perform binary NOT
    void performNot(void)
    {
        mercury::internal::BitsetNot(_arr, ((N + 0x1f) >> 5));
    }

    //! Check if all bits are set to true
    bool testAll(void) const
    {
        return mercury::internal::BitsetTestAll(_arr, ((N + 0x1f) >> 5));
    }

    //! Check if any bits are set to true
    bool testAny(void) const
    {
        return mercury::internal::BitsetTestAny(_arr, ((N + 0x1f) >> 5));
    }

    //! Check if none of the bits are set to true
    bool testNone(void) const
    {
        return mercury::internal::BitsetTestNone(_arr, ((N + 0x1f) >> 5));
    }

    //! Compute the cardinality of a bitset
    size_t cardinality(void) const
    {
        return mercury::internal::BitsetCardinality(_arr, ((N + 0x1f) >> 5));
    }

    //! Extract the bitset to an array
    void extract(size_t base, std::vector<size_t> *out) const
    {
        platform_assert(out);
        mercury::internal::BitsetExtract(_arr, ((N + 0x1f) >> 5), base, out);
    }

    //! Extract the bitset to an array
    void extract(std::vector<size_t> *out) const
    {
        this->extract(0, out);
    }

    //! Compute the and cardinality between two bitsets
    static size_t AndCardinality(const FixedBitset &lhs, const FixedBitset &rhs)
    {
        return mercury::internal::BitsetAndCardinality(lhs._arr, rhs._arr,
                                                       ((N + 0x1f) >> 5));
    }

    //! Compute the andnot cardinality between two bitsets
    static size_t AndnotCardinality(const FixedBitset &lhs,
                                    const FixedBitset &rhs)
    {
        return mercury::internal::BitsetAndnotCardinality(lhs._arr, rhs._arr,
                                                          ((N + 0x1f) >> 5));
    }

    //! Compute the xor cardinality between two bitsets
    static size_t XorCardinality(const FixedBitset &lhs, const FixedBitset &rhs)
    {
        return mercury::internal::BitsetXorCardinality(lhs._arr, rhs._arr,
                                                       ((N + 0x1f) >> 5));
    }

    //! Compute the or cardinality between two bitsets
    static size_t OrCardinality(const FixedBitset &lhs, const FixedBitset &rhs)
    {
        return mercury::internal::BitsetOrCardinality(lhs._arr, rhs._arr,
                                                      ((N + 0x1f) >> 5));
    }

    //! Convert a array pointer to bitset pointer
    static FixedBitset *Cast(uint32_t *arr)
    {
        return reinterpret_cast<FixedBitset<N> *>(arr);
    }

    //! Convert a array pointer to bitset pointer
    static const FixedBitset *Cast(const uint32_t *arr)
    {
        return reinterpret_cast<const FixedBitset<N> *>(arr);
    }

    //! Convert a array pointer to bitset pointer
    static FixedBitset *Cast(uint64_t *arr)
    {
        return reinterpret_cast<FixedBitset<N> *>(arr);
    }

    //! Convert a array pointer to bitset pointer
    static const FixedBitset *Cast(const uint64_t *arr)
    {
        return reinterpret_cast<const FixedBitset<N> *>(arr);
    }

private:
    uint32_t _arr[(N + 0x1f) >> 5];
};

/*! Fixed Bitset Module (Special)
 */
template <>
class FixedBitset<0>
{
public:
    enum
    {
        MAX_SIZE = 0
    };

    //! Retrieve max size of bitset
    constexpr size_t size(void) const
    {
        return MAX_SIZE;
    }
};

} // namespace mercury

#endif // __MERCURY_UTILITY_FIXED_BITSET_H__
