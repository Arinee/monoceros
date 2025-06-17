#ifndef __COMMON_PQ_CODEBOOK_H__
#define __COMMON_PQ_CODEBOOK_H__

#include <iostream>
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <memory>
#include "framework/index_meta.h"

namespace mercury {

class PQCodebook
{
public:
    typedef std::shared_ptr<PQCodebook> Pointer;

    PQCodebook(const mercury::IndexMeta& indexMeta,
               size_t roughCentroidNum, 
               size_t integrateCentroidNum,
               size_t fragmentNum)
        : _indexMeta(indexMeta),
        _roughCentroidNum(roughCentroidNum),
        _integrateCentroidNum(integrateCentroidNum),
        _fragmentNum(fragmentNum)
    {
        _integrateCodebook.resize(_fragmentNum);
        _layerPattern.push_back((uint32_t)_roughCentroidNum); // default one layer
        _layerInfo.push_back((uint32_t)_roughCentroidNum); // default one layer
    };

    ~PQCodebook() {};

    const void *getRoughCentroid(size_t centroidIndex) 
    {
        size_t index = centroidIndex * getRoughCentroidSize();
        if (index >= _roughCodebook.size()) {
            return nullptr;
        }
        return &_roughCodebook[index];
    }

    const void *getRoughCentroid(size_t level, size_t levelIndex) 
    {
        if (level >= _layerInfo.size() || levelIndex >= _layerInfo[level]) {
            return nullptr;
        }
        size_t centroidIndex = 0;
        for (size_t i = 0; i < level; ++i) {
            centroidIndex += _layerInfo[i];
        }
        centroidIndex += levelIndex;
        return getRoughCentroid(centroidIndex);
    }

    // input rough centroid
    bool appendRoughCentroid(const void *roughCentroid, size_t sz) 
    {
        if (sz == 0 || sz != getRoughCentroidSize()) {
            fprintf(stderr, "wrong size. roughCentroid size: %lu\n", sz);
            return false;
        }
        size_t lens = _roughCodebook.size();
        _roughCodebook.resize(lens + sz);
        memcpy(&_roughCodebook[lens], roughCentroid, sz);
        return true;
    }

    // support multi layer of rough centroid
    // such as: 10000*100
    bool setLayerInfo(const std::vector<uint32_t> &layerPattern) 
    {
        std::vector<uint32_t> layerInfo;
        uint32_t value = 1;
        uint32_t totalNum = 0;
        for (int i = 0, sz = layerPattern.size(); i < sz; ++i) {
            value *= layerPattern[i];
            layerInfo.push_back(value);
            totalNum += value;
        }
        // check right
        if (totalNum != _roughCentroidNum) {
            fprintf(stderr, "multi layer info is wrong.\n");
            return false;
        }
        _layerInfo = layerInfo;
        _layerPattern = layerPattern;
        return true;
    }

    const void *getIntegrateCentroid(size_t fragmentIndex, size_t centroidIndex) 
    {
        if (fragmentIndex >= _fragmentNum) {
            return nullptr;
        }
        size_t index = centroidIndex * getIntegrateCentroidSize();
        if (index >= _integrateCodebook[fragmentIndex].size()) {
            return nullptr;
        }
        return &_integrateCodebook[fragmentIndex][index];
    }

    // input integrate centroid
    bool appendIntegrateCentroid(size_t fragmentIndex, const void *integrateCentroid, size_t sz) 
    {
        if (sz == 0 || sz != getIntegrateCentroidSize()) {
            fprintf(stderr, "wrong size, integrateCentroid.size: %lu\n", sz);
            return false;
        }
        if (fragmentIndex >= _fragmentNum) {
            fprintf(stderr, "wrong fragmentIndex for emplaceIntegrate. fragmentIndex: %lu\n", fragmentIndex);
            return false;
        }
        size_t lens = _integrateCodebook[fragmentIndex].size();
        _integrateCodebook[fragmentIndex].resize(lens + sz);
        memcpy(&_integrateCodebook[fragmentIndex][lens], integrateCentroid, sz);
        return true;
    }

    bool checkValid(const mercury::IndexMeta &meta) const 
    {
        if (_indexMeta.type() != meta.type() || 
                _indexMeta.sizeofElement() != meta.sizeofElement()) {
            return false;
        }
        size_t roughSizeInBytes = _roughCentroidNum * getRoughCentroidSize();
        if (_roughCodebook.empty() 
                || _roughCodebook.size() != roughSizeInBytes) {
            return false;
        } 
        size_t integrateSizeInBytes = _integrateCentroidNum * getIntegrateCentroidSize();
        for (size_t i = 0; i < _fragmentNum; ++i) {
            if (_integrateCodebook[i].empty()
                    || _integrateCodebook[i].size() != integrateSizeInBytes) {
                return false;
            }
        }
        return true;
    }

    size_t getRoughCentroidSize() const 
    {
        return _indexMeta.sizeofElement();
    }

    // get layer infomation
    const std::vector<uint32_t> &getLayerInfo() const 
    {
        return _layerInfo;
    }

    // get layer pattern
    const std::vector<uint32_t> &getLayerPattern() const 
    {
        return _layerPattern;
    }

    // integrate Centroid size
    size_t getIntegrateCentroidSize() const 
    {
        return _indexMeta.sizeofElement() / _fragmentNum;
    }

    size_t getDimension() const 
    {
        return _indexMeta.dimension();
    }

    size_t getRoughCentoridNum() const 
    {
        return _roughCentroidNum;
    }

    size_t getIntegrateCentroidNum() const 
    {
        return _integrateCentroidNum;
    }

    size_t getFragmentNum() const 
    {
        return _fragmentNum;
    }

    void setIndexMeta(const mercury::IndexMeta &indexMeta) 
    {
        _indexMeta = indexMeta;
    }

    const mercury::IndexMeta& getIndexMeta(void) 
    {
        return _indexMeta;
    }

private:
    std::vector<uint8_t> _roughCodebook;
    std::vector<std::vector<uint8_t>> _integrateCodebook;
    std::vector<uint32_t> _layerInfo;
    std::vector<uint32_t> _layerPattern;
    mercury::IndexMeta _indexMeta;
    size_t _roughCentroidNum;
    size_t _integrateCentroidNum;
    size_t _fragmentNum;
};

} // namespace mercury

#endif // __COMMON_PQ_CODEBOOK_H__
