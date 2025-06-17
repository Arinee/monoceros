#include "query_distance_matrix1.h"
#include "src/core/framework/index_distance.h"
#include "src/core/utils/string_util.h"

MERCURY_NAMESPACE_BEGIN(core);

QueryDistanceMatrix1::QueryDistanceMatrix1()
    : _queryCodeFeature(nullptr)
    , _withCodeFeature(false)
    , _distanceArray(nullptr)
    , _ipValArray(nullptr)
    , _ipCentArray(nullptr)
    , _fragmentNum(0)
    , _centroidNum(0)
    , _elemSize(0)
{
}

QueryDistanceMatrix1::QueryDistanceMatrix1(const IndexMeta &indexMeta, CentroidResource* centroidResource)
    : _centroidResource(centroidResource)
    , _indexMeta(indexMeta)
    , _fragmentIndexMeta(indexMeta)
    , _queryCodeFeature(nullptr)
    , _withCodeFeature(false)
    , _distanceArray(nullptr)
    , _ipValArray(nullptr)
    , _ipCentArray(nullptr)
{
    auto iMeta = _centroidResource->getIntegrateMeta();
    _fragmentNum = iMeta.fragmentNum;
    _centroidNum = iMeta.centroidNum;
    _elemSize = iMeta.elemSize;
    _fragmentIndexMeta.setDimension(_indexMeta.dimension() / _fragmentNum);
    // _fragmentIndexMeta.setType(IndexMeta::FeatureTypes::kTypeFloat);

    // index meta should be match with product fragment
    assert(_fragmentIndexMeta.sizeofElement() == _elemSize);
}

QueryDistanceMatrix1::~QueryDistanceMatrix1() 
{
    if (_queryCodeFeature) {
        delete[] _queryCodeFeature;
        _queryCodeFeature = nullptr;
    }
    if (_distanceArray) {
        delete[] _distanceArray;
        _distanceArray = nullptr;
    }
    if (_ipValArray) {
        delete[] _ipValArray;
        _ipValArray = nullptr;
    }
    if (_ipCentArray) {
        delete[] _ipCentArray;
        _ipCentArray = nullptr;
    }
}

bool QueryDistanceMatrix1::init(const void *queryFeature, const std::vector<size_t>& levelScanLimit, bool withCodeFeature)
{
    _withCodeFeature = withCodeFeature;

    if (_fragmentIndexMeta.sizeofElement() != _elemSize) {
        std::cerr << "index meta not match with centroid resource, element size: "
            << _fragmentIndexMeta.sizeofElement() << " " << _elemSize << std::endl;
        return false;
    }
    if (!computeDistanceMatrix(queryFeature)) {
        return false;
    }
    return true;
}

bool QueryDistanceMatrix1::initDistanceMatrix(const void *queryFeature, bool withCodeFeature)
{
    _withCodeFeature = withCodeFeature;
    if (_fragmentIndexMeta.sizeofElement() != _elemSize) {
        std:: cerr << "index meta not match with centroid resource, element size: "
            << _fragmentIndexMeta.sizeofElement() << " " <<  _elemSize << std::endl;
        return false;
    }
    if (!computeDistanceMatrix(queryFeature)) {
        return false;
    }
    return true;
}


bool QueryDistanceMatrix1::getQueryCodeFeature(UInt8Vector& queryCodeFeature)
{
    if (!_withCodeFeature) {
        return false;
    }
    for (size_t i = 0; i < _fragmentNum; ++i) {
        queryCodeFeature.push_back(_queryCodeFeature[i]);
    }
    return true;
}

bool QueryDistanceMatrix1::computeCentroid(const void *feature, const std::vector<size_t>& levelScanLimit)
{
    assert(_centroids.empty());
    size_t level = 0;
    auto roughMeta = _centroidResource->getRoughMeta();
    if (levelScanLimit.size() < roughMeta.levelCnt - 1) {
        std::cerr << "rough level scan limit size not match level count "
                << levelScanLimit.size() << " " << roughMeta.levelCnt - 1 << std::endl;
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

// term3: - 2 * (x|y_R)
bool QueryDistanceMatrix1::computeIpValMatrix(const void *feature)
{
    if (_fragmentIndexMeta.sizeofElement() != _elemSize) {
        std:: cerr << "index meta not match with centroid resource, element size: "
            << _fragmentIndexMeta.sizeofElement() << " " <<  _elemSize << std::endl;
        return false;
    }
    if (_ipValArray) {
        delete[] _ipValArray;
        _ipValArray = nullptr;
    }
    _ipValArray = new(std::nothrow) distance_t[_centroidNum * _fragmentNum];
    if (UNLIKELY(!_ipValArray)) {
        std::cerr << "alloc distance array buf error center fragment "
               << _centroidNum <<  " " << _fragmentNum << std::endl;
        return false;
    }

    for (size_t fragmentIndex = 0; fragmentIndex < _fragmentNum ; ++fragmentIndex) {
        const char* fragmentFeature = reinterpret_cast<const char*>(feature) + fragmentIndex * _elemSize;
        for (size_t centroidIndex = 0; centroidIndex < _centroidNum; ++centroidIndex) {
            const void* fragmentCenter = _centroidResource->getValueInIntegrateMatrix(fragmentIndex, centroidIndex);
            size_t index = fragmentIndex * _centroidNum + centroidIndex;
            _ipValArray[index] = 2 * _fragmentIndexMeta.fip(fragmentFeature, fragmentCenter);
        }
    }
    return true;
}

bool QueryDistanceMatrix1::computeDistanceMatrix(const void *feature)
{
    if (UNLIKELY(_withCodeFeature)) {
        _queryCodeFeature = new(std::nothrow) uint8_t[_fragmentNum];
        if (UNLIKELY(!_queryCodeFeature)) {
            std::cerr << "alloc query code feature buf error fragment num "
                   << _fragmentNum << std::endl;
            return false;
        }
    }
    if (_distanceArray) {
        delete[] _distanceArray;
        _distanceArray = nullptr;
    }
    _distanceArray = new(std::nothrow) distance_t[_centroidNum * _fragmentNum];
    if (UNLIKELY(!_distanceArray)) {
        std::cerr << "alloc distance array buf error center fragment "
               << _centroidNum <<  " " << _fragmentNum << std::endl;
        return false;
    }

    for (size_t fragmentIndex = 0; fragmentIndex < _fragmentNum ; ++fragmentIndex) {
        distance_t minDistance = std::numeric_limits<distance_t>::max();
        const char* fragmentFeature = reinterpret_cast<const char*>(feature) + fragmentIndex * _elemSize;
        for (size_t centroidIndex = 0; centroidIndex < _centroidNum; ++centroidIndex) {
            const void* fragmentCenter = _centroidResource->getValueInIntegrateMatrix(fragmentIndex, centroidIndex);
            size_t index = fragmentIndex * _centroidNum + centroidIndex;

            _distanceArray[index] = _fragmentIndexMeta.distance(fragmentFeature, fragmentCenter);
            if (UNLIKELY(_withCodeFeature && _distanceArray[index] < minDistance)) {
                minDistance = _distanceArray[index];
                _queryCodeFeature[fragmentIndex] = static_cast<uint8_t>(centroidIndex);
            }
        }
    }
    return true;
}

MERCURY_NAMESPACE_END(core);
