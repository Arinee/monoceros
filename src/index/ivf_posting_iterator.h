/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     corase_index.h
 *   \author   yunheng@xiaohongshu.com
 *   \date     Jan 2019
 *   \version  1.0.0
 *   \brief    ivf posting
 */

#ifndef __MERCURY_INDEX_IVF_POSTING_ITERATOR_H__
#define __MERCURY_INDEX_IVF_POSTING_ITERATOR_H__

#include "coarse_index.h"
#include "posting_iterator.h"

namespace mercury {

class IvfPostingIterator : public PostingIterator
{
public:
    typedef std::shared_ptr<IvfPostingIterator> Pointer;

    IvfPostingIterator()
        :index_(0)
    {
    }

    IvfPostingIterator(size_t capacity)
        :index_(0)
    {
        ivf_posting_list_.reserve(capacity);
    }

    void push_back(const CoarseIndex::PostingIterator& posting_iterator)
    {
        ivf_posting_list_.push_back(posting_iterator);
    }

    virtual ~IvfPostingIterator() override
    {}

    // 不考虑加锁情况
    virtual docid_t next() override
    {
        return ivf_posting_list_[index_].next();
    }

    virtual bool finish() override
    {
        for (; index_ < ivf_posting_list_.size(); ++index_) {
            if (unlikely(ivf_posting_list_[index_].finish())) {
                //LOG_INFO("index: %lu, finish: %s", index_, "true");
                continue;
            }
            return false;
        }
        return true;
    }

    docid_t next1()
    {
        return ivf_posting_list_[index_].next();
    }

    bool finish1()
    {
        for (; index_ < ivf_posting_list_.size(); ++index_) {
            if (unlikely(ivf_posting_list_[index_].finish())) {
                //LOG_INFO("index: %lu, finish: %s", index_, "true");
                continue;
            }
            return false;
        }
        return true;
    }

private:
    std::vector<CoarseIndex::PostingIterator> ivf_posting_list_;
    size_t index_;
};

} // mercury

#endif //__MERCURY_INDEX_IVF_POSTING_ITERATOR_H__
