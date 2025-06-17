#ifndef __MERCURY_MOCK_INDEX_HOLDER_H__
#define __MERCURY_MOCK_INDEX_HOLDER_H__

#include "framework/vector_holder.h"

namespace mercury {

template <typename T>
struct FeatureTypeTrait {
    const static mercury::VectorHolder::FeatureType value = IndexMeta::kTypeUnknown;
};
template <>
struct FeatureTypeTrait<float> {
    const static mercury::VectorHolder::FeatureType value = IndexMeta::kTypeFloat;
};
template <>
struct FeatureTypeTrait<double> {
    const static mercury::VectorHolder::FeatureType value = IndexMeta::kTypeDouble;
};
template <>
struct FeatureTypeTrait<int32_t> {
    const static mercury::VectorHolder::FeatureType value = IndexMeta::kTypeBinary;
};
template <>
struct FeatureTypeTrait<int8_t> {
    const static mercury::VectorHolder::FeatureType value = IndexMeta::kTypeInt8;
};
template <>
struct FeatureTypeTrait<int16_t> {
    const static mercury::VectorHolder::FeatureType value = IndexMeta::kTypeInt16;
};


/*! Mock Index Holder
 */
template <typename T>
class MockVectorHolder : public mercury::VectorHolder
{
public:
    typedef mercury::VectorHolder VectorHolder;
    typedef std::shared_ptr<MockVectorHolder> Pointer;

public:
    MockVectorHolder() 
        :_keyVec(),
        _valueVec(),
        _labelsVec(),
        _isLabelled(false)
    {}

    /*! Index Holder Iterator
     */
    class Iterator : public VectorHolder::Iterator
    {
    public:
        //! Constructor
        Iterator(const MockVectorHolder *holder) : _index(0), _holder(holder) {}

        //! Retrieve pointer of data
        virtual const void *data(void) const override
        {
            return _holder->_valueVec[_index].data();
        }

        //! Test if the iterator is valid
        virtual bool isValid(void) const override
        {
            return (_index < _holder->_keyVec.size());
        }

        //! Retrieve primary key
        virtual uint64_t key(void) const override
        {
            return _holder->_keyVec[_index];
        }

        //! Retrieve labels
        virtual std::vector<size_t> labels(void) const override
        {
            return _holder->_labelsVec[_index];
        }

        //! Next iterator
        virtual void next(void) override
        {
            ++_index;
        }

        //! Reset the iterator
        virtual void reset(void) override
        {
            _index = 0;
        }

    private:
        size_t _index;
        const MockVectorHolder *_holder;

        //! Disable them
        Iterator(void) = delete;
        Iterator(const Iterator &) = delete;
        Iterator(Iterator &&) = delete;
        Iterator &operator=(const Iterator &) = delete;
    };

    virtual bool hasLabels(void) const override
    {
        return _isLabelled;
    }

    void setLabel(bool yes)
    {
        _isLabelled = yes;
    }

    //! Retrieve count of elements in holder
    virtual size_t count(void) const override
    {
        return _keyVec.size();
    }

    //! Retrieve dimension
    virtual size_t dimension(void) const override
    {
        if (_valueVec.empty()) {
            return 0;
        }
        return _valueVec[0].size();
    }

    //! Retrieve type information
    virtual VectorHolder::FeatureType type(void) const override
    {
        return FeatureTypeTrait<T>::value;
    }

    //! Create a new iterator
    virtual VectorHolder::Iterator::Pointer createIterator(void) const override
    {
        return VectorHolder::Iterator::Pointer(
            new MockVectorHolder::Iterator(this));
    }

    //! Request a change in capacity
    void reserve(size_t size)
    {
        _keyVec.reserve(size);
        _valueVec.reserve(size);
        _labelsVec.reserve(size);
    }

    //! Append an element into holder
    void emplace(uint64_t key, const std::vector<T> &val, const std::vector<size_t> &labels)
    {
        _keyVec.emplace_back(key);
        _valueVec.push_back(val);
        _labelsVec.push_back(labels);
    }

    //! Append an element into holder
    void emplace(uint64_t key, const std::vector<T> &val)
    {
        _keyVec.emplace_back(key);
        _valueVec.push_back(val);
        // push an empty labels
        _labelsVec.push_back(std::vector<size_t>());
    }

private:
    std::vector<uint64_t> _keyVec;
    std::vector<std::vector<T>> _valueVec;
    std::vector<std::vector<size_t>> _labelsVec;
    bool _isLabelled;
};

} // namespace 

#endif // __MERCURY_MOCK_INDEX_HOLDER_H__
