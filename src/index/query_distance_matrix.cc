#include "query_distance_matrix.h"
#include "framework/index_logger.h"
#include "framework/index_distance.h"
#include "utils/string_util.h"

using namespace std;
using namespace mercury;

namespace mercury {

QueryDistanceMatrix::QueryDistanceMatrix()
    : _queryCodeFeature(nullptr)
    , _withCodeFeature(false)
    , _distanceArray(nullptr)
    , _fragmentNum(0)
    , _centroidNum(0)
    , _elemSize(0)
{
}

QueryDistanceMatrix::QueryDistanceMatrix(const IndexMeta &indexMeta, CentroidResource* centroidResource)
    : _centroidResource(centroidResource)
    , _indexMeta(indexMeta)
    , _fragmentIndexMeta(indexMeta)
    , _queryCodeFeature(nullptr)
    , _withCodeFeature(false)
    , _distanceArray(nullptr)
{
    auto iMeta = _centroidResource->getIntegrateMeta();
    _fragmentNum = iMeta.fragmentNum;
    _centroidNum = iMeta.centroidNum;
    _elemSize = iMeta.elemSize;
    _fragmentIndexMeta.setDimension(_indexMeta.dimension() / _fragmentNum);

    // index meta should be match with product fragment
    assert(_fragmentIndexMeta.sizeofElement() == _elemSize);
}

QueryDistanceMatrix::~QueryDistanceMatrix() 
{
    if (_queryCodeFeature) {
        delete[] _queryCodeFeature;
        _queryCodeFeature = nullptr;
    }
    if (_distanceArray) {
        delete[] _distanceArray;
        _distanceArray = nullptr;
    }
}

bool QueryDistanceMatrix::init(const void *queryFeature, const vector<size_t>& levelScanLimit, bool withCodeFeature)
{
    _withCodeFeature = withCodeFeature;

    if (_fragmentIndexMeta.sizeofElement() != _elemSize) {
        LOG_ERROR("index meta not match with centroid resource, element size[%zd]|[%zd]", 
                _fragmentIndexMeta.sizeofElement(), _elemSize);
        return false;
    }
    if (!computeCentroid(queryFeature, levelScanLimit)) {
        return false;
    }
    if (!computeDistanceMatrix(queryFeature)) {
        return false;
    }
    return true;
}

bool QueryDistanceMatrix::initDistanceMatrix(const void *queryFeature)
{
    _withCodeFeature = false;
    if (_fragmentIndexMeta.sizeofElement() != _elemSize) {
        LOG_ERROR("index meta not match with centroid resource, element size[%zd]|[%zd]", _fragmentIndexMeta.sizeofElement(), _elemSize);
        return false;
    }
    if (!computeDistanceMatrix(queryFeature)) {
        return false;
    }
    return true;
}


bool QueryDistanceMatrix::getQueryCodeFeature(UInt16Vector& queryCodeFeature)
{
    if (!_withCodeFeature) {
        return false;
    }
    for (size_t i = 0; i < _fragmentNum; ++i) {
        queryCodeFeature.push_back(_queryCodeFeature[i]);
    }
    return true;
}

bool QueryDistanceMatrix::computeCentroid(const void *feature, const vector<size_t>& levelScanLimit)
{
    assert(_centroids.empty());
    size_t level = 0;
    auto roughMeta = _centroidResource->getRoughMeta();
    if (levelScanLimit.size() < roughMeta.levelCnt - 1) {
        LOG_ERROR("rough level scan limit size[%zd] not match level count[%u]", 
                levelScanLimit.size(), roughMeta.levelCnt - 1);
        return false;
    }
    for (uint32_t i = 0; i < roughMeta.centroidNums[level]; ++i) {
        const void* centroidValue = _centroidResource->getValueInRoughMatrix(level, i); 
        float score = _indexMeta.distance(feature, centroidValue);
        _centroids.emplace(i, score);
    }

    // 0 ~ last, find nearest centroids
    for (level = 1; level < roughMeta.levelCnt; ++level)
    {
        uint32_t centroidNum = roughMeta.centroidNums[level];
        CentroidCandidate candidate;
        candidate.swap(_centroids);

        size_t scanNum = levelScanLimit[level - 1];
        while (!candidate.empty() && scanNum-- > 0) {
            auto doc = candidate.top();
            candidate.pop();
            for (uint32_t i = 0; i < centroidNum; ++i) {
                uint32_t centroid = doc.index * centroidNum + i;
                const void* centroidValue = _centroidResource->getValueInRoughMatrix(level, centroid); 
                float score = _indexMeta.distance(feature, centroidValue);
                _centroids.emplace(centroid, score);
            }
        }
    }

    return true;
}

bool QueryDistanceMatrix::computeDistanceMatrix(const void *feature)
{
    if (unlikely(_withCodeFeature)) {
        _queryCodeFeature = new(nothrow) uint16_t[_fragmentNum];
        if (unlikely(!_queryCodeFeature)) {
            LOG_WARN("alloc query code feature buf error fragment num[%lu] " ,
                   _fragmentNum);
            return false;
        }
    }
    if (_distanceArray) {
        delete[] _distanceArray;
        _distanceArray = nullptr;
    }
    _distanceArray = new(nothrow) distance_t[_centroidNum * _fragmentNum];
    if (unlikely(!_distanceArray)) {
        LOG_WARN("alloc distance array buf error center [%lu] fragment[%lu] " ,
               _centroidNum, _fragmentNum);
        return false;
    }

    for (size_t fragmentIndex = 0; fragmentIndex < _fragmentNum ; ++fragmentIndex) {
        distance_t minDistance = numeric_limits<distance_t>::max();
        const char* fragmentFeature = reinterpret_cast<const char*>(feature) + fragmentIndex * _elemSize;
        for (size_t centroidIndex = 0; centroidIndex < _centroidNum; ++centroidIndex) {
            const void* fragmentCenter = _centroidResource->getValueInIntegrateMatrix(fragmentIndex, centroidIndex);
            size_t index = fragmentIndex * _centroidNum + centroidIndex;
            _distanceArray[index] = _fragmentIndexMeta.distance(fragmentFeature, fragmentCenter);
            if (unlikely(_withCodeFeature && _distanceArray[index] < minDistance)) {
                minDistance = _distanceArray[index];
                _queryCodeFeature[fragmentIndex] = static_cast<uint16_t>(centroidIndex);
            }
        }
    }
    return true;
}

}; // namespace mercury

