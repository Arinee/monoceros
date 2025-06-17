/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     index_reformer.h
 *   \author   Hechong.xyf
 *   \date     Jul 2018
 *   \version  1.0.0
 *   \brief    Interface of Mercury Index Reformer
 */

#ifndef __MERCURY_INDEX_REFORMER_H__
#define __MERCURY_INDEX_REFORMER_H__

#include "utility/thread_pool.h"
#include "vector_holder.h"

namespace mercury { namespace core {

/*! Index Reformer Parameters
 */
using ReformerParams = IndexParams;

/*! Index Feature Reformer
 */
struct IndexFeatureReformer
{
    //! Index Feature Reformer Pointer
    typedef std::shared_ptr<IndexFeatureReformer> Pointer;

    //! Destructor
    virtual ~IndexFeatureReformer(void) {}

    //! Train the reformer with vector
    virtual int train(ThreadPool &pool, const VectorHolder::Pointer &holder) = 0;

    //! Reform the vector
    virtual void reform(const void *feat, size_t feat_size,
                        std::string *out) = 0;

    //! Retrieve the original meta
    virtual const IndexMeta &prometa(void) const = 0;

    //! Retrieve the translate meta
    virtual const IndexMeta &meta(void) const = 0;
};

/*! Index Query Reformer
 */
struct IndexQueryReformer
{
    //! Index Query Reformer Pointer
    typedef std::shared_ptr<IndexQueryReformer> Pointer;

    //! Destructor
    virtual ~IndexQueryReformer(void) {}

    //! Reform the vector
    virtual void reform(const void *feat, size_t feat_size,
                        std::string *out) = 0;

    //! Normalize score
    virtual float normalize(float score) const = 0;
};

/*! Index Reformer
 */
struct IndexReformer
{
    //! Index Reformer Pointer
    typedef std::shared_ptr<IndexReformer> Pointer;

    //! Destructor
    virtual ~IndexReformer(void) {}

    //! Create a feature reformer
    virtual IndexFeatureReformer::Pointer
    createFeatureReformer(const IndexMeta &meta,
                          const ReformerParams &params) const = 0;

    //! Create a query reformer
    virtual IndexQueryReformer::Pointer
    createQueryReformer(const IndexMeta &meta,
                        const ReformerParams &params) const = 0;
};

/*! Index Reformer Holder
 */
class IndexReformerHolder : public VectorHolder
{
public:
    /*! Index Reformer Holder Iterator
     */
    class Iterator : public VectorHolder::Iterator
    {
    public:
        //! Constructor
        Iterator(const IndexReformerHolder *owner)
            : _owner(owner), _iter(owner->_holder->createIterator()), _buffer()
        {
            this->transform();
        }

        //! Destructor
        virtual ~Iterator(void) {}

        //! Retrieve pointer of data
        virtual const void *data(void) const
        {
            return _buffer.data();
        }

        //! Test if the iterator is valid
        virtual bool isValid(void) const
        {
            return _iter->isValid();
        }

        //! Retrieve primary key
        virtual uint64_t key(void) const
        {
            return _iter->key();
        }

        //! Next iterator
        virtual void next(void)
        {
            _iter->next();
            this->transform();
        }

        //! Reset the iterator
        virtual void reset(void)
        {
            _iter->reset();
            this->transform();
        }

        //! Retrieve labels
        virtual std::vector<size_t> labels(void) const
        {
            return _iter->labels();
        }

        //! Transform the record
        void transform(void)
        {
            if (_iter->isValid()) {
                _owner->_reformer->reform(
                    _iter->data(), _owner->_holder->sizeofElement(), &_buffer);
            }
        }

    private:
        const IndexReformerHolder *_owner;
        VectorHolder::Iterator::Pointer _iter;
        std::string _buffer;
    };

    //! Destructor
    virtual ~IndexReformerHolder(void) {}

    //! Retrieve count of elements in holder (-1 indicates unknown)
    virtual size_t count(void) const
    {
        return _holder->count();
    }

    //! Retrieve dimension
    virtual size_t dimension(void) const
    {
        return _reformer->meta().dimension();
    }

    //! Retrieve type information
    virtual IndexMeta::FeatureTypes type(void) const
    {
        return _reformer->meta().type();
    }

    //! Retrieve if it supports labels
    virtual bool hasLabels(void) const
    {
        return _holder->hasLabels();
    }

    //! Create a new iterator
    virtual VectorHolder::Iterator::Pointer createIterator(void) const
    {
        return VectorHolder::Iterator::Pointer(
            new IndexReformerHolder::Iterator(this));
    }

    //! Retrieve size of feature
    virtual size_t sizeofElement(void) const
    {
        return _reformer->meta().sizeofElement();
    }

    //! Proxy the holder with a reformer
    static VectorHolder::Pointer
    Proxy(const VectorHolder::Pointer &holder,
          const IndexFeatureReformer::Pointer &reformer)
    {
        if (!holder || !reformer || !reformer->prometa().isMatched(*holder)) {
            return VectorHolder::Pointer();
        }
        return VectorHolder::Pointer(new IndexReformerHolder(holder, reformer));
    }

protected:
    //! Constructor
    IndexReformerHolder(const VectorHolder::Pointer &holder,
                        const IndexFeatureReformer::Pointer &reformer)
        : _holder(holder), _reformer(reformer)
    {
    }

private:
    //! Disable them
    IndexReformerHolder(void) = delete;
    IndexReformerHolder(const IndexReformerHolder &) = delete;
    IndexReformerHolder(IndexReformerHolder &&) = delete;
    IndexReformerHolder &operator=(const IndexReformerHolder &) = delete;
    IndexReformerHolder &operator=(IndexReformerHolder &&) = delete;

    //! Members
    VectorHolder::Pointer _holder;
    IndexFeatureReformer::Pointer _reformer;
};

} // core
} // namespace mercury

#endif // __MERCURY_INDEX_REFORMER_H__
