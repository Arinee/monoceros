/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     centroid_center.h
 *   \author   yunheng@xiaohongshu.com
 *   \date     Jan 2019
 *   \version  1.0.0
 *   \brief    
 */

#ifndef __MERCURY_INDEX_CENTROID_RESOURCE_H__
#define __MERCURY_INDEX_CENTROID_RESOURCE_H__

#include <iostream>
#include <math.h>
#include <queue>
#include <memory>
#include "framework/index_distance.h"
#include "pq_common.h"

namespace mercury {

/*
 * For use in queue of centroid quantize
 */
struct CentroidInfo {
    uint32_t    index;
    float       dist;
    CentroidInfo(uint32_t i, float d)
        :index(i),
        dist(d)
    {}
    bool operator<(const CentroidInfo &other) const
    {
        return dist < other.dist;
    }
    bool operator>(const CentroidInfo &other) const
    {
        return dist > other.dist;
    }
};

class CentroidResource
{
public:
    typedef std::shared_ptr<CentroidResource> Pointer;

    class RoughMeta
    {
    public:
        RoughMeta()
            : magic(0)
            , elemSize(0)
            , levelCnt(0)
        {}
        RoughMeta(uint32_t elementSize, uint32_t levelCount, std::vector<uint32_t> levelCentroids)
            : magic(0)
            , elemSize(elementSize)
            , levelCnt(levelCount)
            , centroidNums(move(levelCentroids))
        {}
        ~RoughMeta() {}
    
    public:
        uint32_t magic;
        uint32_t elemSize;
        uint32_t levelCnt;
        std::vector<uint32_t> centroidNums;
    };
    
    class IntegrateMeta
    {
    public:
        IntegrateMeta()
            : magic(0)
            , elemSize(0)
            , fragmentNum(0)
            , centroidNum(0)
        {}
        IntegrateMeta(uint32_t elementSize, uint32_t fragmentNumber, uint32_t centroidNumber)
            : magic(0)
            , elemSize(elementSize)
            , fragmentNum(fragmentNumber)
            , centroidNum(centroidNumber)
        {}
        ~IntegrateMeta() {}
    
    public:
        uint32_t magic;
        uint32_t elemSize;
        uint32_t fragmentNum;
        uint32_t centroidNum;
    };

    CentroidResource() : _memAlloc(false)
        , _roughMatrix(nullptr)
        , _integrateMatrix(nullptr)
    {};

    virtual ~CentroidResource();

    bool init(void *roughBase, size_t roughLen, void *integrateBase, size_t integrateLen);
    bool init(void *roughBase, size_t roughLen);

    bool create(const RoughMeta &roughMeta, const IntegrateMeta &integrateMeta);
    bool create(const RoughMeta &roughMeta);

    bool IsInit() const{
        return (_roughOnly && _roughMatrix != nullptr) ||
        (_roughMatrix != nullptr || _integrateMatrix != nullptr);
    }
    

    void dumpRoughMatrix(std::string &roughString) const;
    void dumpIntegrateMatrix(std::string &integrateString) const;

    virtual RoughMeta& getRoughMeta() { return _roughMeta; };
    virtual IntegrateMeta& getIntegrateMeta() { return _integrateMeta; };

    bool setValueInRoughMatrix(size_t level, size_t centroidIndex, const void* value);
    inline const void *getValueInRoughMatrix(size_t level, size_t centroidIndex) const
    {
        if (unlikely(!_roughMatrix)) {
            return nullptr;
        }
        size_t index = 0;
        size_t ratio = 1;
        for (size_t l = 0; l < level; ++l) {
            index += _roughMeta.centroidNums[l] * ratio;
            ratio *= _roughMeta.centroidNums[l];
        }
        index += centroidIndex;
        index *= _roughMeta.elemSize;
        return _roughMatrix + index;
    }
    bool setValueInIntegrateMatrix(size_t fragmentIndex, size_t centroidIndex, const void* value);
    inline const void *getValueInIntegrateMatrix(size_t fragmentIndex, size_t centroidIndex) const
    {
        if (unlikely(!_integrateMatrix)) {
            return nullptr;
        }
        return _integrateMatrix + (fragmentIndex * _integrateMeta.centroidNum + centroidIndex) * _integrateMeta.elemSize;
    }
    
    inline size_t getLeafCentroidNum()
    {
        size_t count = 1;
        for (size_t l = 0; l < _roughMeta.levelCnt; ++l) {
            count *= _roughMeta.centroidNums[l];
        }
        return count;
    }
    
    bool DumpToFile(const std::string rough_name, const std::string& interagte_name);
    bool DumpToFile(const std::string rough_name);

private:
    bool parseRoughContent(char *pBase, size_t fileLength);
    bool parseIntegrateContent(char *pBase, size_t fileLength);
    bool validate() 
    { 
        // TODO Why always true
        return true;//_roughMeta._fragmentNum == _integrateMeta._fragmentNum; 
    };

protected:
    bool _memAlloc = false;
    RoughMeta _roughMeta;
    IntegrateMeta _integrateMeta;
    char *_roughMatrix = nullptr;
    size_t _roughMatrixSize = 0;
    char *_integrateMatrix = nullptr;
    size_t _integrateMatrixSize = 0;
    bool _roughOnly = false;
};

} // namespace mercury

#endif // __MERCURY_INDEX_CENTROID_RESOURCE_H__
