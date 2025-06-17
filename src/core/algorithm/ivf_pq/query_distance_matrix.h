#pragma once

#include "../centroid_resource.h"
#include "src/core/common/common.h"
#include "src/core/common/pq_common.h"
#include "src/core/framework/index_meta.h"
#include <assert.h>
#include <memory>

MERCURY_NAMESPACE_BEGIN(core);

class QueryDistanceMatrix
{
public:
    typedef std::shared_ptr<QueryDistanceMatrix> Pointer;
    typedef std::priority_queue<CentroidInfo, std::vector<CentroidInfo>, std::greater<CentroidInfo>> CentroidCandidate;

    QueryDistanceMatrix();
    QueryDistanceMatrix(const IndexMeta &indexMeta, CentroidResource *centroidResource);
    ~QueryDistanceMatrix();

public:
    bool init(const void *queryFeature, const std::vector<size_t> &levelScanLimit, bool withCodeFeature = false);
    bool initDistanceMatrix(const void *queryFeature, bool withCodeFeature = false);
    // only for build and label
    bool getQueryCodeFeature(UInt16Vector &queryCodeFeature);

    void *GetDistanceArray()
    {
        return _distanceArray;
    }

    distance_t getDistance(size_t centroidIndex, size_t fragmentIndex) const
    {
        assert(fragmentIndex < _fragmentNum);
        assert(centroidIndex < _centroidNum);
        return _distanceArray[fragmentIndex * _centroidNum + centroidIndex];
    }

    CentroidCandidate &getCentroids()
    {
        return _centroids;
    }

    size_t getFragmentNum() const
    {
        return _fragmentNum;
    }

    size_t getCentroidNum() const
    {
        return _centroidNum;
    }

    void setWithCodeFeature(bool withCodeFeature)
    {
        this->_withCodeFeature = withCodeFeature;
    }

    bool computeCentroid(const void *feature, const std::vector<size_t> &levelScanLimit);
    bool computeDistanceMatrix(const void *feature);
    bool initQueryCodeFeature(void);

private:
    CentroidResource *_centroidResource;
    IndexMeta _indexMeta;
    IndexMeta _fragmentIndexMeta;
    uint16_t *_queryCodeFeature;
    CentroidCandidate _centroids;
    bool _withCodeFeature;

protected:
    distance_t *_distanceArray;
    size_t _fragmentNum;
    size_t _centroidNum;
    size_t _elemSize;
};

MERCURY_NAMESPACE_END(core);
