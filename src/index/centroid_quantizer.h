/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     level1_quantizer.h
 *   \author   yunheng@xiaohongshu.com
 *   \date     Jan 2019
 *   \version  1.0.0
 *   \brief    L1 Quantizer
 */

#ifndef __MERCURY_INDEX_L1_QUANTIZER_H__
#define __MERCURY_INDEX_L1_QUANTIZER_H__

#include <vector>
#include "index.h"
#include "centroid_resource.h"

namespace mercury {

#if 0
// L1 Qiamtizer Holder
class Level1Quantizer
{
public:
    ///function interface
    virtual ~Level1Quantizer() {};

    /// search and return labels
    virtual std::vector<uint32_t> Search(const void *val, size_t len, size_t nprobe, 
            IndexMeta* index_meta) = 0;

    virtual bool CreateLevelOneQuantizer(std::map<std::string, size_t>& stab);
    virtual bool LoadLevelOneQuantizer(IndexPackage &package);
    virtual bool DumpLevelOneQuantizer(IndexPackage &package);

    /// calc L1 label
    virtual int CalcLabel(const void *val, size_t len,
            IndexMeta* index_meta);
protected :
    IndexMeta index_meta_;
};
#endif

class CentroidQuantizer
{
public:
    CentroidQuantizer(){
        centroid_resource_ = CentroidResource::Pointer(new CentroidResource);
    }

    /// get and set centroid 
    void set_centroid_resource(CentroidResource::Pointer centroid_resource);
    CentroidResource* get_centroid_resource();

    /// get rough centroid count
    size_t get_slot_num();

    /// clac label
    int32_t CalcLabel(const void *val, size_t len,
            IndexMeta* index_meta, const std::vector<size_t>& roughLevelScanLimit);
    /// virtual function implement
    std::vector<uint32_t> Search(const void *val, size_t len, size_t nprobe, 
            const std::vector<size_t>& levelScanLimit, IndexMeta* index_meta);

    bool DumpLevelOneQuantizer(IndexPackage &package);
    bool CreateLevelOneQuantizer(std::map<std::string, size_t>& stab);
    bool LoadLevelOneQuantizer(IndexPackage &package);
protected:
    /// centroid centers
    CentroidResource::Pointer centroid_resource_;

    /// TODO remove it , use centroid resource buff
    std::string roughMatrix;
    std::string integrateMatrix;
};

} // namespace mercury

#endif //__MERCURY_INDEX_L1_QUANTIZER_H__
