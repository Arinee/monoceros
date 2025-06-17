/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     bitset_helper.h
 *   \author   Hechong.xyf
 *   \date     Aug 2018
 *   \version  1.0.0
 *   \brief    Interface of Mercury Utility Flat Bitset
 */

#ifndef __MERCURY_UTILITY_BITSET_HELPER_H__
#define __MERCURY_UTILITY_BITSET_HELPER_H__

#include "internal/bitset.h"

namespace mercury {

/*! Bitset Helper
 */
class BitsetHelper
{
public:
    //! Constructor
    BitsetHelper(void) : _arr(nullptr), _size(0u) {}

    //! Constructor
    BitsetHelper(void *buf, size_t len)
        : _arr(reinterpret_cast<uint32_t *>(buf)), _size(len / sizeof(uint32_t))
    {
    }

    //! Mount a buffer as bitset
    void mount(void *buf, size_t len)
    {
        _arr = reinterpret_cast<uint32_t *>(buf);
        _size = len / sizeof(uint32_t);
    }

    //! Umount the buffer
    void umount(void)
    {
        _arr = nullptr;
        _size = 0u;
    }

    //！Clear the bitset
    void clear(void)
    {
        memset(_arr, 0, sizeof(uint32_t) * _size);
    }

    //! Test a bit in bitset
    bool test(size_t num) const
    {
        platform_assert_with((_size << 5) > num, "overflow argument");
        return ((_arr[num >> 5] & (1u << (num & 0x1f))) != 0);
    }

    //! Set a bit in bitset
    void set(size_t num)
    {
        platform_assert_with((_size << 5) > num, "overflow argument");
        uint32_t mask = (1u << (num & 0x1f));
        _arr[num >> 5] |= mask;
    }

    //! Reset a bit in bitset
    void reset(size_t num)
    {
        platform_assert_with((_size << 5) > num, "overflow argument");
        uint32_t mask = (1u << (num & 0x1f));
        _arr[num >> 5] &= ~mask;
    }

    //! Toggle a bit in bitset
    void flip(size_t num)
    {
        platform_assert_with((_size << 5) > num, "overflow argument");
        uint32_t mask = (1u << (num & 0x1f));
        _arr[num >> 5] ^= mask;
    }

    //! Check if all bits are set to true
    bool testAll(void) const
    {
        return mercury::internal::BitsetTestAll(_arr, _size);
    }

    //! Check if any bits are set to true
    bool testAny(void) const
    {
        return mercury::internal::BitsetTestAny(_arr, _size);
    }

    //! Check if none of the bits are set to true
    bool testNone(void) const
    {
        return mercury::internal::BitsetTestNone(_arr, _size);
    }

    //! Compute the cardinality of a bitset
    size_t cardinality(void) const
    {
        return mercury::internal::BitsetCardinality(_arr, _size);
    }

    //! Extract the bitset to an array
    void extract(size_t base, std::vector<size_t> *out) const
    {
        platform_assert(out);
        mercury::internal::BitsetExtract(_arr, _size, base, out);
    }

    //! Extract the bitset to an array
    void extract(std::vector<size_t> *out) const
    {
        this->extract(0, out);
    }

    void* getBase() const
    {
        return _arr;
    }

    //! Calculate the size of buffer if it contains N bits
    static size_t CalcBufferSize(size_t N)
    {
        return (((N + 0x1f) >> 5) << 2);
    }

    //! Calculate the count of bits can be contained
    static size_t CalcBitsCount(size_t len)
    {
        return ((len >> 2) << 2);
    }

private:
    uint32_t *_arr;
    size_t _size;
};

} // namespace mercury

#endif // __MERCURY_UTILITY_BITSET_HELPER_H__
