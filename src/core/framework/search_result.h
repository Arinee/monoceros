/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     search_result.h
 *   \author   qiuming
 *   \date     Jan 2019
 *   \version  1.0.0
 *   \brief    result of search request
 */

#ifndef __MERCURY_SEARCH_RESULT_H__
#define __MERCURY_SEARCH_RESULT_H__

#include <stdint.h>
#include <stdlib.h>
#include "src/core/common/common.h"

MERCURY_NAMESPACE_BEGIN(core);

/*! Result of search request
*/
struct SearchResult
{
    uint64_t key;   //! Primary Key
    // TODO  gloid type
    uint64_t gloid; //! Internal gloid
    float score;    //! Distance Score
    uint32_t poolId = 0;

    //! Constructor
    SearchResult(void) : key(0), gloid(0), score(0.0f) {}

    //! Constructor
    SearchResult(uint64_t k, uint32_t i, float v, uint32_t poolId = 0) : key(k), gloid(i), score(v), poolId(poolId)
    {
    }

    //! Constructor
    SearchResult(const SearchResult &rhs)
        : key(rhs.key), gloid(rhs.gloid), score(rhs.score), poolId(rhs.poolId)
    {
    }

    //! Assignment
    SearchResult &operator=(const SearchResult &rhs)
    {
        key = rhs.key;
        gloid = rhs.gloid;
        score = rhs.score;
        poolId = rhs.poolId;
        return *this;
    }

    //! equal
    bool operator==(const SearchResult &rhs) const
    {
        return key == rhs.key && gloid == rhs.gloid
        && abs(score - rhs.score) <= 1e-6;
    }

    //! Less than
    bool operator<(const SearchResult &rhs) const
    {
        return (this->score < rhs.score);
    }

    //! Greater than
    bool operator>(const SearchResult &rhs) const
    {
        return (this->score > rhs.score);
    }

};


MERCURY_NAMESPACE_END(core);
#endif // __MERCURY_SEARCH_RESULT_H__

