/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     index_ivf.h
 *   \author   yunheng@xiaohongshu.com
 *   \date     Jan 2019
 *   \version  1.0.0
 *   \brief    Index Ivf Abstract
 */

#ifndef __MERCURY_INDEX_INDEX_IVF_H__
#define __MERCURY_INDEX_INDEX_IVF_H__

#include "index.h"
#include "centroid_quantizer.h"

namespace mercury {

class IndexIvf : public Index
{
public:
    typedef std::shared_ptr<IndexIvf> Pointer;
    IndexIvf(){
        _centroidQuantizer = new CentroidQuantizer;
    }

    virtual ~IndexIvf() 
    {
        DELETE_AND_SET_NULL(_centroidQuantizer);
    }

    // load profile data
    bool LoadIVFIndexFromPackage(IndexPackage &package);
    bool DumpIVFIndexToPackage(IndexPackage &package, bool only_dump_meta);
    bool CreateIVFIndexFromPackage(std::map<std::string, size_t>& stab);
    CentroidQuantizer* get_centroid_quantizer();
    CentroidResource* getCentroidResource()
    {
        return _centroidQuantizer->get_centroid_resource();
    }
protected:
    // CentroidQuantizer holder
    CentroidQuantizer *_centroidQuantizer = nullptr;	    
};
} // namespace mercury

#endif // __MERCURY_INDEX_INDEX_IVF_H__
