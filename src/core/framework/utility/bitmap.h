/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     bitmap.h
 *   \author   Hechong.xyf
 *   \date     Dec 2017
 *   \version  1.0.0
 *   \brief    Interface of Mercury Utility Bitmap
 */

#ifndef __MERCURY_UTILITY_BITMAP_H__
#define __MERCURY_UTILITY_BITMAP_H__

#include "fixed_bitset.h"
#include <algorithm>
#include <vector>

namespace mercury {

/*! Bitmap Module
 */
class Bitmap
{
public:
    typedef FixedBitset<65536u> Bucket;

    //! Constructor
    Bitmap(void) : _arr() {}

    //! Constructor
    Bitmap(const Bitmap &rhs)
    {
        this->copy(rhs);
    }

    //! Destructor
    ~Bitmap(void)
    {
        this->clear();
    }

    //! Assignment
    Bitmap &operator=(const Bitmap &rhs)
    {
        this->copy(rhs);
        return *this;
    }

    //! Retrieve bucket size of bitmap
    size_t getBucketSize(void) const
    {
        return _arr.size();
    }

    //！Clear the bitmap
    void clear(void);

    //! Copy the content from another bitmap
    void copy(const Bitmap &rhs);

    //! Remove the none buckets
    void shrinkToFit(void);

    //! Test a bit in bitmap
    bool test(size_t num) const;

    //! Set a bit in bitmap
    void set(size_t num);

    //! Reset a bit in bitmap
    void reset(size_t num);

    //! Toggle a bit in bitmap
    void flip(size_t num);

    //! Perform binary AND
    void performAnd(const Bitmap &rhs);

    //! Perform binary AND NOT
    void performAndnot(const Bitmap &rhs);

    //! Perform binary OR
    void performOr(const Bitmap &rhs);

    //! Perform binary XOR
    void performXor(const Bitmap &rhs);

    //! Perform binary NOT (It will expand the whole map)
    void performNot(void);

    //! Check if all bits are set to true
    bool testAll(void) const;

    //! Check if any bits are set to true
    bool testAny(void) const;

    //! Check if none of the bits are set to true
    bool testNone(void) const;

    //! Compute the cardinality of a bitmap
    size_t cardinality(void) const;

    //! Extract the bitmap to an array
    void extract(size_t base, std::vector<size_t> *out) const;

    //! Extract the bitmap to an array
    void extract(std::vector<size_t> *out) const
    {
        this->extract(0, out);
    }

    //! Compute the and cardinality between two bitmaps
    static size_t AndCardinality(const Bitmap &lhs, const Bitmap &rhs);

    //! Compute the andnot cardinality between two bitmaps
    static size_t AndnotCardinality(const Bitmap &lhs, const Bitmap &rhs);

    //! Compute the xor cardinality between two bitmaps
    static size_t XorCardinality(const Bitmap &lhs, const Bitmap &rhs);

    //! Compute the or cardinality between two bitmaps
    static size_t OrCardinality(const Bitmap &lhs, const Bitmap &rhs);

private:
    std::vector<Bucket *> _arr;
};

} // namespace mercury

#endif // __MERCURY_UTILITY_BITMAP_H__
