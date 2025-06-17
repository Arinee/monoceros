#ifndef __MERCURY_VECS_FILE_HOLDER_H__
#define __MERCURY_VECS_FILE_HOLDER_H__

#include "framework/vector_holder.h"
#include "vecs_reader.h"
#include <string>

namespace mercury {

/*! 
 *  framework will use VectorHolder in this way:
 *  for (iter = createIterator(); iter && iter->isValid(); iter->next()) {
 *      key = iter->key();
 *      data = iter->data();
 *  }p
 */

class VecsFileHolder : public VectorHolder
{
public:
    typedef std::shared_ptr<VecsFileHolder> Pointer;

    explicit VecsFileHolder(const IndexMeta &meta, bool catEnabled = false)
            : _indexMeta(meta),
              _vecsReader(meta.sizeofElement(), catEnabled)
    {}

    bool load(const std::string& filePath)
    {
        return _vecsReader.load(filePath);
    }

    const mercury::IndexMeta &indexMeta(void) const
    {
        return _indexMeta;
    }

    /*! 
     * Vector Holder Iterator
     */
    class Iterator : public VectorHolder::Iterator
    {
    public:
        //! Constructor
        explicit Iterator(const VecsFileHolder &holder)
                : _cursor(0),
                  _vecsReader(holder._vecsReader)
        {
        }

        //! Test if the iterator is valid
        bool isValid(void) const override
        {
            return _cursor < _vecsReader.numVecs();
        }

        //! Retrieve primary key
        key_t key(void) const override
        {
            return _vecsReader.getKey(_cursor);
        }

        //! Retrieve pointer of data
        const void *data() const override
        {
            return _vecsReader.getVector(_cursor);
        }

        cat_t cat() const override
        {
            return _vecsReader.getCat(_cursor);
        }

        //! Next iterator
        void next(void) override
        {
            ++_cursor;
        }

        //! Reset the iterator
        void reset(void) override
        {
            _cursor = 0;
        }

    private:
        size_t _cursor;
        const VecsReader &_vecsReader;
    };

    VectorHolder::Iterator::Pointer createIterator(void) const override
    {
        // make sure iter has value when createIterator finished
        VectorHolder::Iterator::Pointer iter(new VecsFileHolder::Iterator(*this));
        return iter;
    }

    //! Retrieve count of elements in holder
    size_t count(void) const override
    {
        return _vecsReader.numVecs();
    }

    //! Retrieve dimension
    size_t dimension(void) const override
    {
        return _indexMeta.dimension();
    }

    //! Retrieve type information
    VectorHolder::FeatureType type(void) const override
    {
        return _indexMeta.type();
    }

private:
    IndexMeta _indexMeta;
    VecsReader _vecsReader;
};

} // namespace mercury

#endif // __MERCURY_VECS_FILE_HOLDER_H__
