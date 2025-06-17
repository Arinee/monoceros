/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     heap.h
 *   \author   Hechong.xyf
 *   \date     Jul 2018
 *   \version  1.0.0
 *   \brief    Interface of Heap adapter
 */

#ifndef __MERCURY_UTILITY_HEAP_H__
#define __MERCURY_UTILITY_HEAP_H__

#include <algorithm>
#include <functional>
#include <limits>
#include <utility>
#include <vector>

namespace mercury {

/*! Heap Adapter
 */
template <typename T, typename TCompare = std::less<T>,
          typename TBase = std::vector<T>>
class Heap : public TBase
{
public:
    //! Constructor
    Heap(void) : TBase(), _limit(std::numeric_limits<size_t>::max()), _compare()
    {
    }

    //! Constructor
    Heap(size_t max) : TBase(), _limit(std::max<size_t>(max, 1u)), _compare()
    {
        TBase::reserve(_limit);
    }

    //! Constructor
    Heap(const Heap &rhs) : TBase(rhs), _limit(rhs._limit), _compare() {}

    //! Constructor
    Heap(Heap &&rhs) : TBase(std::move(rhs)), _limit(rhs._limit), _compare() {}

    //! Constructor
    Heap(const TBase &rhs)
        : TBase(rhs), _limit(std::numeric_limits<size_t>::max()), _compare()
    {
        std::make_heap(TBase::begin(), TBase::end(), _compare);
    }

    //! Constructor
    Heap(TBase &&rhs)
        : TBase(std::move(rhs)), _limit(std::numeric_limits<size_t>::max()),
          _compare()
    {
        std::make_heap(TBase::begin(), TBase::end(), _compare);
    }

    //! Assignment
    Heap &operator=(const Heap &rhs)
    {
        TBase::operator=(static_cast<const TBase &>(rhs));
        _limit = rhs._limit;
        return *this;
    }

    //! Assignment
    Heap &operator=(Heap &&rhs)
    {
        TBase::operator=(std::move(static_cast<TBase &&>(rhs)));
        _limit = rhs._limit;
        return *this;
    }

    //! Exchange the content
    void swap(Heap &rhs)
    {
        TBase::swap(static_cast<TBase &>(rhs));
        std::swap(_limit, rhs._limit);
    }

    //! Pop the front element
    void pop(void)
    {
        if (TBase::size() > 1) {
            auto last = TBase::end() - 1;
            this->replaceHeap(TBase::begin(), last, std::move(*last));
        }
        TBase::pop_back();
    }

    //! Insert a new element into the heap
    template <class... TArgs>
    void emplace(TArgs &&... args)
    {
        if (this->full()) {
            typename std::remove_reference<T>::type val(
                std::forward<TArgs>(args)...);

            auto first = TBase::begin();
            if (_compare(val, *first)) {
                this->replaceHeap(first, TBase::end(), std::move(val));
            }
        } else {
            TBase::emplace_back(std::forward<TArgs>(args)...);
            std::push_heap(TBase::begin(), TBase::end(), _compare);
        }
    }

    //! Insert a new element into the heap
    void push(const T &val)
    {
        if (this->full()) {
            auto first = TBase::begin();
            if (_compare(val, *first)) {
                this->replaceHeap(first, TBase::end(), val);
            }
        } else {
            TBase::push_back(val);
            std::push_heap(TBase::begin(), TBase::end(), _compare);
        }
    }

    //! Insert a new element into the heap
    void push(T &&val)
    {
        if (this->full()) {
            auto first = TBase::begin();
            if (_compare(val, *first)) {
                this->replaceHeap(first, TBase::end(), std::move(val));
            }
        } else {
            TBase::push_back(std::move(val));
            std::push_heap(TBase::begin(), TBase::end(), _compare);
        }
    }

    //! Retrieve the limit of heap
    size_t limit(void) const
    {
        return _limit;
    }

    //! Limit the heap with max size
    void limit(size_t max)
    {
        _limit = std::max<size_t>(max, 1u);
        TBase::reserve(_limit);
    }

    //! Unlimit the size of heap
    void unlimit(void)
    {
        _limit = std::numeric_limits<size_t>::max();
    }

    //! Check whether the heap is full
    bool full(void) const
    {
        return (TBase::size() == _limit);
    }

    //! Update the heap
    void update(void)
    {
        std::make_heap(TBase::begin(), TBase::end(), _compare);
        while (_limit < TBase::size()) {
            this->pop();
        }
    }

    //! Sort the elements in the heap
    void sort(void)
    {
        std::sort(TBase::begin(), TBase::end(), _compare);
    }

protected:
    //! Replace the top element of heap
    template <typename TRandomIterator, typename TValue>
    void replaceHeap(TRandomIterator first, TRandomIterator last, TValue &&val)
    {
        using _DistanceType =
            typename std::iterator_traits<TRandomIterator>::difference_type;

        _DistanceType hole = 0;
        _DistanceType count = _DistanceType(last - first);

        if (count > 1) {
            _DistanceType child = (hole << 1) + 1;

            while (child < count) {
                _DistanceType right_child = child + 1;

                if (right_child < count &&
                    _compare(*(first + child), *(first + right_child))) {
                    child = right_child;
                }
                if (!_compare(val, *(first + child))) {
                    break;
                }
                *(first + hole) = std::move(*(first + child));
                hole = child;
                child = (hole << 1) + 1;
            }
        }
        *(first + hole) = std::forward<TValue>(val);
    }

private:
    size_t _limit;
    TCompare _compare;
};

/*! Key Value Heap Comparer
 */
template <typename TKey, typename TValue, typename TCompare = std::less<TValue>>
struct KeyValueHeapComparer
{
    //! Function call
    bool operator()(const std::pair<TKey, TValue> &lhs,
                    const std::pair<TKey, TValue> &rhs) const
    {
        return _compare(lhs.second, rhs.second);
    }

private:
    TCompare _compare;
};

/*! Key Value Heap
 */
template <typename TKey, typename TValue, typename TCompare = std::less<TValue>>
using KeyValueHeap =
    Heap<std::pair<TKey, TValue>, KeyValueHeapComparer<TKey, TValue, TCompare>>;

} // namespace mercury

#endif //__MERCURY_UTILITY_HEAP_H__
