/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     bitmap.cc
 *   \author   Hechong.xyf
 *   \date     Dec 2017
 *   \version  1.0.0
 *   \brief    Implementation of Mercury Utility Bitmap
 */

#include "bitmap.h"

namespace mercury {

void Bitmap::clear(void)
{
    for (std::vector<Bucket *>::iterator iter = _arr.begin();
         iter != _arr.end(); ++iter) {
        delete (*iter);
    }
    _arr.clear();
}

void Bitmap::copy(const Bitmap &rhs)
{
    this->clear();

    for (std::vector<Bucket *>::const_iterator iter = rhs._arr.begin();
         iter != rhs._arr.end(); ++iter) {
        Bucket *bucket = NULL;
        if (*iter) {
            bucket = new Bucket(*(*iter));
        }
        _arr.push_back(bucket);
    }
}

void Bitmap::shrinkToFit(void)
{
    size_t shrink_count = 0;
    std::vector<Bucket *>::reverse_iterator iter;

    for (iter = _arr.rbegin(); iter != _arr.rend(); ++iter) {
        if (*iter) {
            if (!(*iter)->testNone()) {
                break;
            }
            delete (*iter);
            *iter = NULL;
        }
        ++shrink_count;
    }
    for (; iter != _arr.rend(); ++iter) {
        if ((*iter) && (*iter)->testNone()) {
            delete (*iter);
            *iter = NULL;
        }
    }
    if (shrink_count != 0) {
        _arr.resize(_arr.size() - shrink_count);
    }
}

bool Bitmap::test(size_t num) const
{
    // High 16 bits
    size_t offset = num >> 16;

    if (offset < _arr.size()) {
        const Bucket *bucket = _arr[offset];
        if (bucket) {
            // Low 16 bits
            return bucket->test(static_cast<uint16_t>(num));
        }
    }
    return false;
}

void Bitmap::set(size_t num)
{
    // High 16 bits
    size_t offset = num >> 16;
    if (offset >= _arr.size()) {
        _arr.resize(offset + 1, NULL);
    }

    Bucket *&bucket = _arr[offset];
    if (!bucket) {
        bucket = new Bucket;
    }
    // Low 16 bits
    bucket->set(static_cast<uint16_t>(num));
}

void Bitmap::reset(size_t num)
{
    // High 16 bits
    size_t offset = num >> 16;
    if (offset >= _arr.size()) {
        _arr.resize(offset + 1, NULL);
    }

    if (offset < _arr.size()) {
        Bucket *bucket = _arr[offset];
        if (bucket) {
            // Low 16 bits
            bucket->reset(static_cast<uint16_t>(num));
        }
    }
}

void Bitmap::flip(size_t num)
{
    // High 16 bits
    uint16_t offset = num >> 16;
    if (offset >= _arr.size()) {
        _arr.resize(offset + 1, NULL);
    }

    Bucket *&bucket = _arr[offset];
    if (!bucket) {
        bucket = new Bucket;
    }
    // Low 16 bits
    bucket->flip(static_cast<uint16_t>(num));
}

void Bitmap::performAnd(const Bitmap &rhs)
{
    size_t overlap = std::min(_arr.size(), rhs._arr.size());

    for (size_t i = 0; i < overlap; ++i) {
        Bucket *&dst = _arr[i];

        if (dst) {
            const Bucket *src = rhs._arr[i];
            if (src) {
                dst->performAnd(*src);
            } else {
                delete dst;
                dst = NULL;
            }
        }
    }
    for (size_t i = overlap; i < _arr.size(); ++i) {
        Bucket *&dst = _arr[i];
        delete dst;
        dst = NULL;
    }
}

void Bitmap::performAndnot(const Bitmap &rhs)
{
    size_t overlap = std::min(_arr.size(), rhs._arr.size());

    for (size_t i = 0; i < overlap; ++i) {
        Bucket *&dst = _arr[i];

        if (dst) {
            const Bucket *src = rhs._arr[i];
            if (src) {
                dst->performAndnot(*src);
            }
        }
    }
}

void Bitmap::performOr(const Bitmap &rhs)
{
    size_t overlap = std::min(_arr.size(), rhs._arr.size());

    for (size_t i = 0; i < overlap; ++i) {
        const Bucket *src = rhs._arr[i];

        if (src) {
            Bucket *&dst = _arr[i];

            if (dst) {
                dst->performOr(*src);
            } else {
                dst = new Bucket(*src);
            }
        }
    }
    for (size_t i = overlap; i < rhs._arr.size(); ++i) {
        const Bucket *src = rhs._arr[i];
        Bucket *bucket = NULL;

        if (src) {
            bucket = new Bucket(*src);
        }
        _arr.push_back(bucket);
    }
}

void Bitmap::performXor(const Bitmap &rhs)
{
    size_t overlap = std::min(_arr.size(), rhs._arr.size());

    for (size_t i = 0; i < overlap; ++i) {
        const Bucket *src = rhs._arr[i];

        if (src) {
            Bucket *&dst = _arr[i];

            if (dst) {
                dst->performXor(*src);
            } else {
                dst = new Bucket(*src);
            }
        }
    }
    for (size_t i = overlap; i < rhs._arr.size(); ++i) {
        const Bucket *src = rhs._arr[i];
        Bucket *bucket = NULL;

        if (src) {
            bucket = new Bucket(*src);
        }
        _arr.push_back(bucket);
    }
}

void Bitmap::performNot(void)
{
    for (std::vector<Bucket *>::iterator iter = _arr.begin();
         iter != _arr.end(); ++iter) {
        Bucket *&bucket = *iter;
        if (!bucket) {
            bucket = new Bucket;
        }
        bucket->performNot();
    }
}

bool Bitmap::testAll(void) const
{
    if (_arr.size() == 0) {
        return false;
    }
    for (std::vector<Bucket *>::const_iterator iter = _arr.begin();
         iter != _arr.end(); ++iter) {
        if (!(*iter) || !(*iter)->testAll()) {
            return false;
        }
    }
    return true;
}

bool Bitmap::testAny(void) const
{
    for (std::vector<Bucket *>::const_iterator iter = _arr.begin();
         iter != _arr.end(); ++iter) {
        if (*iter && (*iter)->testAny()) {
            return true;
        }
    }
    return false;
}

bool Bitmap::testNone(void) const
{
    for (std::vector<Bucket *>::const_iterator iter = _arr.begin();
         iter != _arr.end(); ++iter) {
        if (*iter && !(*iter)->testNone()) {
            return false;
        }
    }
    return true;
}

size_t Bitmap::cardinality(void) const
{
    size_t result = 0;
    for (std::vector<Bucket *>::const_iterator iter = _arr.begin();
         iter != _arr.end(); ++iter) {
        if (*iter) {
            result += (*iter)->cardinality();
        }
    }
    return result;
}

void Bitmap::extract(size_t base, std::vector<size_t> *out) const
{
    for (std::vector<Bucket *>::const_iterator iter = _arr.begin();
         iter != _arr.end(); ++iter) {
        if (*iter) {
            (*iter)->extract(base, out);
        }
        base += Bucket::MAX_SIZE;
    }
}

size_t Bitmap::AndCardinality(const Bitmap &lhs, const Bitmap &rhs)
{
    size_t overlap = std::min(lhs._arr.size(), rhs._arr.size());
    size_t dist = 0;

    for (size_t i = 0; i < overlap; ++i) {
        const Bucket *l = lhs._arr[i];
        const Bucket *r = rhs._arr[i];

        if (l && r) {
            dist += Bucket::AndCardinality(*l, *r);
        }
    }
    return dist;
}

size_t Bitmap::AndnotCardinality(const Bitmap &lhs, const Bitmap &rhs)
{
    size_t overlap = std::min(lhs._arr.size(), rhs._arr.size());
    size_t dist = 0;

    for (size_t i = 0; i < overlap; ++i) {
        const Bucket *l = lhs._arr[i];
        if (l) {
            const Bucket *r = rhs._arr[i];
            if (r) {
                dist += Bucket::AndnotCardinality(*l, *r);
            } else {
                dist += l->cardinality();
            }
        }
    }
    for (size_t i = overlap; i < lhs._arr.size(); ++i) {
        const Bucket *l = lhs._arr[i];
        if (l) {
            dist += l->cardinality();
        }
    }
    return dist;
}

size_t Bitmap::XorCardinality(const Bitmap &lhs, const Bitmap &rhs)
{
    size_t overlap = std::min(lhs._arr.size(), rhs._arr.size());
    size_t dist = 0;

    for (size_t i = 0; i < overlap; ++i) {
        const Bucket *l = lhs._arr[i];
        const Bucket *r = rhs._arr[i];

        if (l && r) {
            dist += Bucket::XorCardinality(*l, *r);
        } else if (l) {
            dist += l->cardinality();
        } else if (r) {
            dist += r->cardinality();
        }
    }
    for (size_t i = overlap; i < lhs._arr.size(); ++i) {
        const Bucket *l = lhs._arr[i];
        if (l) {
            dist += l->cardinality();
        }
    }
    for (size_t i = overlap; i < rhs._arr.size(); ++i) {
        const Bucket *r = rhs._arr[i];
        if (r) {
            dist += r->cardinality();
        }
    }
    return dist;
}

size_t Bitmap::OrCardinality(const Bitmap &lhs, const Bitmap &rhs)
{
    size_t overlap = std::min(lhs._arr.size(), rhs._arr.size());
    size_t dist = 0;

    for (size_t i = 0; i < overlap; ++i) {
        const Bucket *l = lhs._arr[i];
        const Bucket *r = rhs._arr[i];

        if (l && r) {
            dist += Bucket::OrCardinality(*l, *r);
        } else if (l) {
            dist += l->cardinality();
        } else if (r) {
            dist += r->cardinality();
        }
    }
    for (size_t i = overlap; i < lhs._arr.size(); ++i) {
        const Bucket *l = lhs._arr[i];
        if (l) {
            dist += l->cardinality();
        }
    }
    for (size_t i = overlap; i < rhs._arr.size(); ++i) {
        const Bucket *r = rhs._arr[i];
        if (r) {
            dist += r->cardinality();
        }
    }
    return dist;
}

} // namespace mercury
