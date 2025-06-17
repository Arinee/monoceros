/// Copyright (c) 2019, xiaohongshu Inc. All rights reserved.
/// Author: qiuming <qiuming@xiaohongshu.com>
/// Created: 2019-09-01 17:11

#pragma once

#include <stdlib.h>
#include <stdint.h>
#include <algorithm>
#include <vector>
#include "src/core/common/pq_common.h"

MERCURY_NAMESPACE_BEGIN(core);

template <typename HeapNode>
class MyHeap {
public:
    MyHeap(size_t capacity) 
        : _capacity(capacity),
          _fetch_pos(0)
    {
        if (_capacity == 0) {
            _capacity = 500; 
        }
        _heap.reserve(_capacity);
    }

    bool empty(void) const 
    {
        return _heap.empty();
    }

    void makeHeap(void) 
    {
        std::make_heap(_heap.begin(), _heap.end());
    }

    bool isFull(void) const 
    {
        return _heap.size() == _capacity;
    }

    const HeapNode* fetch() {
        if (_fetch_pos < _heap.size()) {
            return &_heap.at(_fetch_pos++);
        }

        return nullptr;
    }

    bool fetchEnd() const {
        return _fetch_pos >= _heap.size();
    }

    size_t getFetchPos() const {
        return _fetch_pos;
    }

    void push(HeapNode&& node) 
    {
        if (_heap.size() < _capacity) {
            _heap.push_back(node);
            std::push_heap(_heap.begin(), _heap.end());
            return;
        }

        if (node < _heap[0]) {
            adjustTop2Down(node);
        } else {
            // this node dont fit,igonre
        }
    }

    void push(const HeapNode& node) 
    {
        if (_heap.size() < _capacity) {
            _heap.push_back(node);
            std::push_heap(_heap.begin(), _heap.end());
            return;
        }

        if (node < _heap[0]) {
            adjustTop2Down(node);
        } else {
            // this node dont fit,igonre
        }
    }

    void pop(void) 
    {
        std::pop_heap(_heap.begin(), _heap.end());
        _heap.pop_back();
    }

    void sort(void) 
    {
        std::sort_heap(_heap.begin(), _heap.end());
    }

    void sortByDocId() {
        std::sort(_heap.begin(), _heap.end(), [](const HeapNode& a, const HeapNode& b) {
                                                  return a.key < b.key;
                                              });
    }

    const HeapNode& top(void) const 
    {
        return _heap[0];
    }

    const std::vector<HeapNode>& getData(void) const
    {
        return _heap;
    }

private:
    void adjustTop2Down(const HeapNode& _x)
    {
        typedef typename std::vector<HeapNode>::difference_type difference_type;
        auto _first = _heap.begin();
        auto _last = _heap.end();
        auto _holeIndex = difference_type(0);
        auto _secondChild = 2 * _holeIndex + 2;
        auto _len = difference_type(_last - _first);

        while (_secondChild < _len)
        {
            if (*(_first + _secondChild) < *(_first + (_secondChild - 1)))
            {
                _secondChild--;
            }
            if (LIKELY(*(_first + _secondChild) < _x))
            {
                *(_first + _holeIndex) = _x;
                return;
            }

            *(_first + _holeIndex) = *(_first + _secondChild);

            _holeIndex = _secondChild;
            _secondChild = 2 * _secondChild + 2;
        }
        if (_secondChild == _len)
        {
            _secondChild--;
            if (UNLIKELY(_x < *(_first + _secondChild)))
            {
                *(_first + _holeIndex) = *(_first + _secondChild);
                _holeIndex = _secondChild;
            }
        }

        *(_first + _holeIndex) = _x;
    }

private:
    std::vector<HeapNode> _heap;
    size_t _capacity;
    size_t _fetch_pos;
};

MERCURY_NAMESPACE_END(core);
