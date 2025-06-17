#ifndef MERCURY_HAMMING_CONTAINER_H_
#define MERCURY_HAMMING_CONTAINER_H_

#include <vector>
#include <cstring>
#include <string>

namespace mercury
{

template <typename T>
class HammingContainer
{
public:
    HammingContainer(void);
    HammingContainer(int maxDist, int keepCntPerDist, int maxReturnSize);
    HammingContainer(const HammingContainer &container);
    HammingContainer(HammingContainer &&container) noexcept;
    ~HammingContainer(void);

public:
    HammingContainer &operator=(HammingContainer &&container) noexcept;

public:
    void push(T node, int dist);
    bool pop(T &node);
    bool pop(T &node, int &dist);
    //reset it before reuse
    void reset();
    void reset(int maxDist, int keepCntPerDist, int maxReturnSize);
    int size()
    { return _size; }
    bool empty()
    { return _size == 0; }

private:
    void init(int maxDist, int keepCntPerDist, int maxReturnSize);

private:
    int _maxDist;
    int _keepCntPerDist;
    int _maxReturnSize;

    int _size;
    //how many elements in current bucket, every distane corresponding to a bucket
    int *_curNum;
    //when pop element, identify next idx in current bucket
    int *_nextIdx;
    //pop in current bucket
    int _curDist;
    std::vector<std::vector<T> > _container;
};

template <typename T>
inline HammingContainer<T>::HammingContainer(void)
{
    _maxDist = 0;
    _keepCntPerDist = 0;
    _maxReturnSize = 0;
    _size = 0;
    _curNum = nullptr;
    _nextIdx = nullptr;
    _curDist = 0;
}

template <typename T>
inline HammingContainer<T>::HammingContainer(int maxDist, int keepCntPerDist, int maxReturnSize)
{
    init(maxDist, keepCntPerDist, maxReturnSize);
}

template <typename T>
inline HammingContainer<T>::HammingContainer(const HammingContainer &container)
    : _container(container._container)
{
    _maxDist = container._maxDist;
    _keepCntPerDist = container._keepCntPerDist;
    _maxReturnSize = container._maxReturnSize;
    _size = container._size;
    _curNum = new int[_maxDist + 1];
    memcpy(_curNum, container._curNum, sizeof(int) * (_maxDist + 1));
    _nextIdx = new int[_maxDist + 1];
    memset(_nextIdx, container._nextIdx, sizeof(int) * (_maxDist + 1));
    _curDist = container._curDist;
}

template <typename T>
inline HammingContainer<T>::HammingContainer(HammingContainer &&container) noexcept
    : _container(std::move(container._container))
{
    _maxDist = container._maxDist;
    _keepCntPerDist = container._keepCntPerDist;
    _maxReturnSize = container._maxReturnSize;
    _size = container._size;
    _curNum = container._curNum;
    _nextIdx = container._nextIdx;
    _curDist = container._curDist;
    
    container._maxDist = 0;
    container._keepCntPerDist = 0;
    container._maxReturnSize = 0;
    container._size = 0;
    container._curNum = nullptr;
    container._nextIdx = nullptr;
    container._curDist = 0;
}

template <typename T>
HammingContainer<T>::~HammingContainer()
{
    delete [] _curNum;
    _curNum = NULL;
    delete [] _nextIdx;
    _nextIdx = NULL;
}

template <typename T>
HammingContainer<T> &HammingContainer<T>::operator=(HammingContainer &&container) noexcept
{
    if (this != &container) {
        delete [] _curNum;
        delete [] _nextIdx;

        _maxDist = container._maxDist;
        _keepCntPerDist = container._keepCntPerDist;
        _maxReturnSize = container._maxReturnSize;
        _size = container._size;
        _curNum = container._curNum;
        _nextIdx = container._nextIdx;
        _curDist = container._curDist;
        _container = std::move(container._container);

        container._maxDist = 0;
        container._keepCntPerDist = 0;
        container._maxReturnSize = 0;
        container._size = 0;
        container._curNum = nullptr;
        container._nextIdx = nullptr;
        container._curDist = 0;
    }

    return *this;
}

template <typename T>
inline void HammingContainer<T>::push(T node, int dist)
{
    //dist exceed max distance, do nothing
    if (dist > _maxDist) {
        return;
    }

    //if current bucket is full, do nothing
    if (_curNum[dist] >= _keepCntPerDist) {
        return;
    }

    _container[dist][_curNum[dist]] = node;
    _curNum[dist]++;
    _size++;

    if (_size > _maxReturnSize) {
        _size = _maxReturnSize;
    }

    return;
}

template <typename T>
inline bool HammingContainer<T>::pop(T &node)
{
    int distance = 0;
    
    return pop(node, distance);
}

template <typename T>
inline bool HammingContainer<T>::pop(T &node, int &dist)
{
    if (_size == 0) {
        return false;
    }

    //find first bucket which can pop
    while ((_nextIdx[_curDist] >= _curNum[_curDist]) && (_curDist <= _maxDist)) {
        _curDist++;
    }

    //if container is empty
    if (_curDist > _maxDist) {
        return false;
    }

    node = _container[_curDist][_nextIdx[_curDist]];
    dist = _curDist;

    _nextIdx[_curDist]++;
    _size--;

    return true;
}

template <typename T>
inline void HammingContainer<T>::reset()
{
    _size = 0;
    memset(_curNum, 0, sizeof(int) * (_maxDist + 1));
    memset(_nextIdx, 0, sizeof(int) * (_maxDist + 1));
    _curDist = 0;    
}

template <typename T>
inline void HammingContainer<T>::reset(int maxDist, 
                                       int keepCntPerDist, 
                                       int maxReturnSize)
{
    if ((_maxDist >= maxDist) && (_keepCntPerDist >= keepCntPerDist)) {
        reset();
        return;
    }

    //release resource
    delete [] _curNum;
    delete [] _nextIdx;
    _container.clear();

    init(maxDist, keepCntPerDist, maxReturnSize);
}

template <typename T>
inline void HammingContainer<T>::init(int maxDist, 
                                      int keepCntPerDist, 
                                      int maxReturnSize)
{
    _maxDist = maxDist;
    _keepCntPerDist = keepCntPerDist;
    _maxReturnSize = maxReturnSize;
    _size = 0;
    _curNum = new int[_maxDist + 1];
    memset(_curNum, 0, sizeof(int) * (_maxDist + 1));
    _nextIdx = new int[_maxDist + 1];
    memset(_nextIdx, 0, sizeof(int) * (_maxDist + 1));
    _curDist = 0;

    _container.resize(_maxDist + 1);
    for (size_t i = 0; i < _container.size(); i++) {
        _container[i].resize(_keepCntPerDist);
    }

    return;
}
} //mercury

#endif //MERCURY_HAMMING_CONTAINER_H_
