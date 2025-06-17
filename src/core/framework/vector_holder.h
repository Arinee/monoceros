/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     vector_holder.h
 *   \author   qiuming
 *   \date     Jan 2019
 *   \version  1.0.0
 *   \brief    Interface of mercury Vector Holder
 */

#ifndef __MERCURY_VECTOR_HOLDER_H__
#define __MERCURY_VECTOR_HOLDER_H__

#include "index_meta.h"
#include "index_params.h"
#include "src/core/common/common_define.h"
#include <memory>
#include <utility>
#include <vector>

MERCURY_NAMESPACE_BEGIN(core);
/*! Vector Holder
 */
struct VectorHolder
{
    //! Vector Holder Pointer
    typedef std::shared_ptr<VectorHolder> Pointer;

    //! Feature Types
    typedef IndexMeta::FeatureTypes FeatureType;

    /*! Vector Holder Iterator
     */
    struct Iterator
    {
        //! Vector Holder Iterator Pointer
        typedef std::unique_ptr<Iterator> Pointer;

        //! Destructor
        virtual ~Iterator(void) {}

        //! Retrieve pointer of data
        virtual const void *data(void) const = 0;

        //! Test if the iterator is valid
        virtual bool isValid(void) const = 0;

        //! Retrieve primary key
        virtual key_t key(void) const = 0;

        //! Retrieve cat id
        virtual cat_t cat(void) const {
            return INVALID_CAT_ID;
        }

        //! Next iterator
        virtual void next(void) = 0;

        //! Reset the iterator
        virtual void reset(void) = 0;

        //! Retrieve labels
        virtual std::vector<size_t> labels(void) const
        {
            return std::vector<size_t>();
        }
    };

    //! Destructor
    virtual ~VectorHolder(void) {}

    //! Retrieve count of elements in holder (-1 indicates unknown)
    virtual size_t count(void) const = 0;

    //! Retrieve dimension
    virtual size_t dimension(void) const = 0;

    //! Retrieve type information
    virtual FeatureType type(void) const = 0;

    //! Retrieve if it supports labels
    virtual bool hasLabels(void) const
    {
        return false;
    }

    //! Create a new iterator
    virtual VectorHolder::Iterator::Pointer createIterator(void) const
    {
        return VectorHolder::Iterator::Pointer();
    }

    //! Retrieve size of feature
    virtual size_t sizeofElement(void) const
    {
        return IndexMeta::Sizeof(this->type(), this->dimension());
    }
};

MERCURY_NAMESPACE_END(core);
#endif // __MERCURY_VECTOR_HOLDER_H__
