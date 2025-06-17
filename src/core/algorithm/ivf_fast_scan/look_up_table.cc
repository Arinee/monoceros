#include "look_up_table.h"
#include "src/core/framework/index_distance.h"
#include "src/core/utils/string_util.h"

MERCURY_NAMESPACE_BEGIN(core);

LoopUpTable::LoopUpTable()
    : _queryCodeFeature(nullptr)
    , _withCodeFeature(false)
    , _distanceArray(nullptr)
    , _qdistanceArray(nullptr)
    , _normalizers(nullptr)
    , _fragmentNum(0)
    , _centroidNum(0)
    , _elemSize(0)
{
}

LoopUpTable::LoopUpTable(const IndexMeta &indexMeta, CentroidResource* centroidResource)
    : _centroidResource(centroidResource)
    , _indexMeta(indexMeta)
    , _fragmentIndexMeta(indexMeta)
    , _queryCodeFeature(nullptr)
    , _withCodeFeature(false)
    , _distanceArray(nullptr)
    , _qdistanceArray(nullptr)
    , _normalizers(nullptr)
{
    auto iMeta = _centroidResource->getIntegrateMeta();
    _fragmentNum = iMeta.fragmentNum;
    _centroidNum = iMeta.centroidNum;
    _elemSize = iMeta.elemSize;
    _fragmentIndexMeta.setDimension(_indexMeta.dimension() / _fragmentNum);

    // index meta should be match with product fragment
    assert(_fragmentIndexMeta.sizeofElement() == _elemSize);
}

LoopUpTable::~LoopUpTable() 
{
    if (_queryCodeFeature) {
        delete[] _queryCodeFeature;
        _queryCodeFeature = nullptr;
    }
    if (_distanceArray) {
        delete[] _distanceArray;
        _distanceArray = nullptr;
    }
    if (_qdistanceArray) {
        delete[] _qdistanceArray;
        _qdistanceArray = nullptr;
    }
    if (_normalizers) {
        delete[] _normalizers;
        _normalizers = nullptr;
    }
}

bool LoopUpTable::init(const void *queryFeature, const std::vector<size_t>& levelScanLimit, bool withCodeFeature)
{
    _withCodeFeature = withCodeFeature;

    if (_fragmentIndexMeta.sizeofElement() != _elemSize) {
        std::cerr << "index meta not match with centroid resource, element size: "
            << _fragmentIndexMeta.sizeofElement() << " " << _elemSize << std::endl;
        return false;
    }
    if (!computeLoopUpTable(queryFeature)) {
        return false;
    }
    return true;
}

bool LoopUpTable::initLoopUpTable(const void *queryFeature, bool withCodeFeature)
{
    _withCodeFeature = withCodeFeature;
    if (_fragmentIndexMeta.sizeofElement() != _elemSize) {
        std:: cerr << "index meta not match with centroid resource, element size: "
            << _fragmentIndexMeta.sizeofElement() << " " <<  _elemSize << std::endl;
        return false;
    }
    if (!computeLoopUpTable(queryFeature)) {
        return false;
    }
    return true;
}


bool LoopUpTable::getQueryCodeFeature(UInt8Vector& queryCodeFeature)
{
    if (!_withCodeFeature) {
        return false;
    }
    for (size_t i = 0; i < _fragmentNum; ++i) {
        queryCodeFeature.push_back(_queryCodeFeature[i]);
    }
    return true;
}

bool LoopUpTable::computeCentroid(const void *feature, const std::vector<size_t>& levelScanLimit)
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

bool LoopUpTable::computeLoopUpTable(const void *feature)
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

bool LoopUpTable::quantizeLoopUpTable() {
    
    if (UNLIKELY(!_distanceArray)) {
        std::cerr << "alloc distance array buf error center fragment "
                << _centroidNum <<  " " << _fragmentNum << std::endl;
        return false;
    }

    // _normalizers[0] = a (scale)
    // _normalizers[1] = b (bias/base)
    _normalizers = new(std::nothrow) float_t[2];

    float a, b;
    std::vector<float> mins(_fragmentNum);
    float max_span_LUT = -HUGE_VAL, max_span_dis = 0;
    b = 0;
    for (size_t i = 0; i < _fragmentNum; i++) {
        mins[i] = tab_min(_distanceArray + i * _centroidNum, _centroidNum);
        float span = tab_max(_distanceArray + i * _centroidNum, _centroidNum) - mins[i];
        max_span_LUT = std::max(max_span_LUT, span);
        max_span_dis += span;
        b += mins[i];
    }
    a = std::min(255 / max_span_LUT, 65535 / max_span_dis);

    _normalizers[0] = a;
    _normalizers[1] = b;

    _qdistanceArray = new(std::nothrow) q_distance_t[_centroidNum * _fragmentNum];
    if (UNLIKELY(!_qdistanceArray)) {
        std::cerr << "alloc quantized distance array buf error center fragment "
               << _centroidNum <<  " " << _fragmentNum << std::endl;
        return false;
    }

    for (size_t i = 0; i < _fragmentNum; i++) {
        round_tab(_distanceArray + i * _centroidNum, _centroidNum, a, mins[i], _qdistanceArray + i * _centroidNum);
    }

    return true;
}

MERCURY_NAMESPACE_END(core);
