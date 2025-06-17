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

namespace mercury 
{

/*! Result of search request
*/
struct SearchResult
{
    uint64_t key;   //! Primary Key
    // TODO  gloid type
    uint64_t gloid; //! Internal gloid
    float score;    //! Distance Score
    uint64_t cat = 0;
    uint32_t poolId = 0;

    //! Constructor
    SearchResult(void) : key(0), gloid(0), score(0.0f) {}

    //! Constructor
    SearchResult(uint64_t k, uint32_t i, float v, uint64_t cat_ = 0) : key(k), gloid(i), score(v), cat(cat_)
    {
    }

    //! Constructor
    SearchResult(const SearchResult &rhs)
        : key(rhs.key), gloid(rhs.gloid), score(rhs.score), cat(rhs.cat)
    {
    }

    //! Assignment
    SearchResult &operator=(const SearchResult &rhs)
    {
        key = rhs.key;
        gloid = rhs.gloid;
        score = rhs.score;
        cat = rhs.cat;
        return *this;
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


} // namespace mercury

#endif // __MERCURY_SEARCH_RESULT_H__

