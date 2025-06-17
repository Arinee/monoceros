#ifndef MERCURY_HEAP_H_
#define MERCURY_HEAP_H_

#include <inttypes.h>
#include <functional>

namespace mercury
{

template <typename T, typename U, typename Compare = std::less<U> >
class Heap
{
public:
    //mount memory or reset, if reset, heap owns memory
    Heap(void);
    //own memory
    Heap(int k);
    Heap(const Heap &heap);
    Heap(Heap &&heap) noexcept;
    ~Heap(void);

public:
    Heap &operator=(Heap &&heap) noexcept;

public:
    //heap is not responsible for memory, the mount guy should take care
    //if you like, you can mount any times, but if you want use reset, unmount first
    inline void mount(T *key, U *value, int k, int usedSize = 0);
    inline void unmount(void);

    inline void top(T &key, U &value);
    inline T topKey(void);
    inline U topValue(void);
    inline void peep(const T *&key, const U *&value, int &size) const;
    inline void peep(T *&key, U *&value, int &size) const;
    inline void peepKey(T *&key, int &size);
    void push(const T &key, const U &value);
    inline void popAndPush(const T &key, const U &value);
    inline void pop(void);
    //may reallocate memory, don't use with mount
    inline void reset(int k = -1);
    inline int capacity(void) const;
    //max size for heap
    inline int heapSize(void) const;
    //number of elements in heap
    inline int size(void) const;
    inline void setSize(int size);
    inline bool empty(void) const;
    inline bool full(void) const;
    inline void order(void);
    inline void orderTop(T &key, U &value);
    inline void orderPop(void);
private:
    void adjustHeap(int size);
    int getTableIndex(uint32_t index);
    inline void init(int k);
    inline void swap(int i, int j);
private:
    Compare _comp;

    bool _mounted;
    int _capacity;
    //top k/bottom k
    int _k;
    //current number of elements
    int _size;
    //only for order pop 
    int _orderIdx;

    //business data
    T *_key;
    T *_baseKey;

    //data be used by heap
    U *_value;
    U *_baseValue;

    const static int MAX_CAPACITY = 65535;
    const static int MAX_TABLE_INDEX = 14;
    const static int LEAF_TABLE[MAX_TABLE_INDEX];
};

template<typename T, typename U, typename Compare>
const int Heap<T, U, Compare>::MAX_CAPACITY;

template<typename T, typename U, typename Compare>
const int Heap<T, U, Compare>::MAX_TABLE_INDEX;

template<typename T, typename U, typename Compare>
const int Heap<T, U, Compare>::LEAF_TABLE[MAX_TABLE_INDEX] = 
{
    1, 2, 4, 8, 16, 32, 64,
    128, 256, 512, 1024, 2048, 4196, 8192
};

template<typename T, typename U, typename Compare>
Heap<T, U, Compare>::Heap(void)
    : _comp(Compare())
{
    _mounted = false;
    _capacity = 0;
    _k = 0;
    _size = 0;
    _orderIdx = 0;
    _key = nullptr;
    _baseKey = nullptr;
    _value = nullptr;
    _baseValue = nullptr;
}

template<typename T, typename U, typename Compare>
Heap<T, U, Compare>::Heap(int k)
    :_comp(Compare())
{
    if (k > MAX_CAPACITY) {
        k = MAX_CAPACITY;
    }

    init(k);
}

template<typename T, typename U, typename Compare>
Heap<T, U, Compare>::Heap(const Heap &heap)
    : _comp(Compare())
{
    _mounted = heap._mounted;
    _capacity = heap._capacity;;
    _k = heap._k;
    _size = heap._size;;
    _orderIdx = heap._orderIdx;;
    _key = nullptr;
    _baseKey = nullptr;
    _value = nullptr;
    _baseValue = nullptr;

    if (_capacity > 0) {
        _key = new T[_capacity];
        _baseKey = _key - 1;
        _value = new U[_capacity];
        _baseValue = _value - 1;
    }
}

template<typename T, typename U, typename Compare>
Heap<T, U, Compare>::Heap(Heap &&heap) noexcept
    : _comp(Compare())
{
    _mounted = heap._mounted;
    _capacity = heap._capacity;;
    _k = heap._k;
    _size = heap._size;;
    _orderIdx = heap._orderIdx;;
    _key = heap._key;
    _baseKey = heap._baseKey;
    _value = heap._value;
    _baseValue = heap._baseValue;

    //reset heap
    heap._mounted = false;
    heap._capacity = 0;
    heap._k = 0;
    heap._size = 0;
    heap._orderIdx = 0;
    heap._key = nullptr;
    heap._baseKey = nullptr;
    heap._value = nullptr;
    heap._baseValue = nullptr;
}

template<typename T, typename U, typename Compare>
Heap<T, U, Compare> &Heap<T, U, Compare>::operator=(Heap &&heap) noexcept
{
    if (this != &heap) {
        if (!_mounted) {
            delete [] _key;
            delete [] _value;
        }

        _mounted = heap._mounted;
        _capacity = heap._capacity;;
        _k = heap._k;
        _size = heap._size;;
        _orderIdx = heap._orderIdx;;
        _key = heap._key;
        _baseKey = heap._baseKey;
        _value = heap._value;
        _baseValue = heap._baseValue;

        //reset heap
        heap._mounted = false;
        heap._capacity = 0;
        heap._k = 0;
        heap._size = 0;
        heap._orderIdx = 0;
        heap._key = nullptr;
        heap._baseKey = nullptr;
        heap._value = nullptr;
        heap._baseValue = nullptr;
    }

    return *this;
}

template<typename T, typename U, typename Compare>
Heap<T, U, Compare>::~Heap(void)
{
    if (!_mounted) {
        delete [] _key;
        delete [] _value;
    }

    _key = nullptr;
    _baseKey = nullptr;

    _value = nullptr;
    _baseValue = nullptr;
}

template<typename T, typename U, typename Compare>
inline void Heap<T, U, Compare>::mount(T *key, U *value, int k, int usedSize)
{
    _mounted = true;
    _capacity = k;
    _k = k;
    _size = usedSize;
    _orderIdx = 0;
    _key = key;
    _baseKey = _key - 1;
    _value = value;
    _baseValue = _value - 1;

    return;
}

template<typename T, typename U, typename Compare>
inline void Heap<T, U, Compare>::unmount(void)
{
    _mounted = false;
    _capacity = 0;
    _k = 0;
    _size = 0;
    _orderIdx = 0;
    _key = nullptr;
    _baseKey = nullptr;
    _value = nullptr;
    _baseValue = nullptr;

    return;
}

template<typename T, typename U, typename Compare>
inline void Heap<T, U, Compare>::top(T &key, U &value)
{
    key = _key[0];
    value = _value[0];
    return;
}

template<typename T, typename U, typename Compare>
inline T Heap<T, U, Compare>::topKey(void)
{
    return _key[0];
}

template<typename T, typename U, typename Compare>
inline U Heap<T, U, Compare>::topValue(void)
{
    return _value[0];
}

template<typename T, typename U, typename Compare>
inline void Heap<T, U, Compare>::peep(const T *&key, const U *&value, int &cnt) const
{
    key = _key;
    value = _value;
    cnt = _size;
}

template<typename T, typename U, typename Compare>
inline void Heap<T, U, Compare>::peep(T *&key, U *&value, int &cnt) const
{
    key = _key;
    value = _value;
    cnt = _size;
}

template<typename T, typename U, typename Compare>
inline void Heap<T, U, Compare>::peepKey(T *&key, int &cnt)
{
    key = _key;
    cnt = _size;
}

template<typename T, typename U, typename Compare>
void Heap<T, U, Compare>::push(const T &key, const U &value)
{
    if (_k == _size) {
        if (_comp(_value[0], value)) {
            return;
        } else {
            return popAndPush(key, value);
        }
    }
    //heap is not full
    uint32_t pos = _size + 1;
    while (pos > 1) {
        uint32_t fatherPos = pos >> 1;
        if (!_comp(_baseValue[fatherPos], value)) {
            break;
        }

        _baseKey[pos] = _baseKey[fatherPos];
        _baseValue[pos] = _baseValue[fatherPos];

        pos = fatherPos;
    }

    _baseKey[pos] = key;
    _baseValue[pos] = value;

    _size++;
    return;
}

template<typename T, typename U, typename Compare>
inline void Heap<T, U, Compare>::popAndPush(const T &key, const U &value)
{
    _key[0] = key;
    _value[0] = value;

    return adjustHeap(_size);
}

template<typename T, typename U, typename Compare>
inline void Heap<T, U, Compare>::pop(void)
{
    _key[0] = _key[_size - 1];
    _value[0] = _value[_size - 1];

    adjustHeap(--_size);

    return;
}

template<typename T, typename U, typename Compare>
inline void Heap<T, U, Compare>::reset(int k)
{
    if (k == -1) {
        _size = 0;
        _orderIdx = 0;
        return;
    }

    if (k > MAX_CAPACITY) {
        k = MAX_CAPACITY;
    }

    if (_capacity >= k) {
        _k = k;
        _size = 0;
        _orderIdx = 0;
        return;
    }

    delete [] _key;
    delete [] _value;

    init(k);

    return;
}

template<typename T, typename U, typename Compare>
inline int Heap<T, U, Compare>::capacity(void) const
{
    return _capacity;
}

template<typename T, typename U, typename Compare>
inline int Heap<T, U, Compare>::heapSize(void) const
{
    return _k;
}

template<typename T, typename U, typename Compare>
inline int Heap<T, U, Compare>::size(void) const
{
    return _size;
}

template<typename T, typename U, typename Compare>
inline void Heap<T, U, Compare>::setSize(int newSize)
{
    _size = newSize;
}

template<typename T, typename U, typename Compare>
inline bool Heap<T, U, Compare>::empty(void) const
{
    return _size == 0;
}

template<typename T, typename U, typename Compare>
inline bool Heap<T, U, Compare>::full(void) const
{
    return _size == _k;
}

template<typename T, typename U, typename Compare>
inline void Heap<T, U, Compare>::order(void)
{
    for (int i = _size - 1; i > 0; i--) {
        swap(0, i);
        adjustHeap(i);
    }
    return;
}

template<typename T, typename U, typename Compare>
inline void Heap<T, U, Compare>::orderTop(T &key, U &value)
{
    key = _key[_orderIdx];
    value = _value[_orderIdx];
    return;
}

template<typename T, typename U, typename Compare>
inline void Heap<T, U, Compare>::orderPop(void)
{
    _orderIdx++;
    _size--;
}

template<typename T, typename U, typename Compare>
void Heap<T, U, Compare>::adjustHeap(int lastPos)
{
    T targetKey = _key[0];
    U targetValue = _value[0];

    //index begin with 1
    uint32_t pos = 1;
    uint32_t child = pos << 1;
    while (child <= static_cast<uint32_t>(lastPos)) {
        uint32_t rightChild = child + 1;
        //should walk to right child
        if ((rightChild <= static_cast<uint32_t>(lastPos) && 
             _comp(_baseValue[child], _baseValue[rightChild]))) {
            child = rightChild;
        }

        //if find the right position 
        if (!_comp(targetValue, _baseValue[child])) {
            break;
        }

        //continue to scan
        _baseValue[pos] = _baseValue[child];
        _baseKey[pos] = _baseKey[child];
        pos = child;
        child = pos << 1;
    }

    _baseValue[pos] = targetValue;
    _baseKey[pos] = targetKey;
    
    return;
}

template<typename T, typename U, typename Compare>
int Heap<T, U, Compare>::getTableIndex(uint32_t index)
{
    uint32_t step = index;

    return static_cast<int>(step);
}

template<typename T, typename U, typename Compare>
inline void Heap<T, U, Compare>::init(int k)
{
    _mounted = false;
    _capacity = k;
    _k = k;
    _size = 0;
    _orderIdx = 0;
    _key = new T[k];
    _baseKey = _key - 1;
    _value = new U[k];
    _baseValue = _value - 1;    

    return;
}

template<typename T, typename U, typename Compare>
inline void Heap<T, U, Compare>::swap(int i, int j)
{
    T key = _key[i];
    U value = _value[i];

    _key[i] = _key[j];
    _value[i] = _value[j];

    _key[j] = key;
    _value[j] = value;

    return;
}
}//end namespace mercury
#endif //MERCURY_HEAP_H_
