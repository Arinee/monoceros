#ifndef __MERCURY_INDEX_QUERY_DISTANCE_MATRIX_H
#define __MERCURY_INDEX_QUERY_DISTANCE_MATRIX_H

#include "pq_common.h"
#include "centroid_resource.h"
#include "framework/search_result.h"
#include "framework/index_meta.h"
#include <memory>
#include <assert.h>

namespace mercury {

class QueryDistanceMatrix
{
public:
    typedef std::shared_ptr<QueryDistanceMatrix> Pointer;
    typedef std::priority_queue<
        CentroidInfo, 
        std::vector<CentroidInfo>, 
        std::greater<CentroidInfo>> CentroidCandidate;

    QueryDistanceMatrix();
    QueryDistanceMatrix(const mercury::IndexMeta &indexMeta, CentroidResource* centroidResource);
    ~QueryDistanceMatrix();

public:
    bool init(const void *queryFeature, const std::vector<size_t> &levelScanLimit, bool withCodeFeature = false);
    bool initDistanceMatrix(const void *queryFeature);
    // only for build and label
    bool getQueryCodeFeature(UInt16Vector& queryCodeFeature);

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
    
    void setWithCodeFeature(bool withCodeFeature){
        this->_withCodeFeature = withCodeFeature;
    }

    bool computeCentroid(const void *feature, const std::vector<size_t>& levelScanLimit);
    bool computeDistanceMatrix(const void *feature);
    bool initQueryCodeFeature(void);

private:
    CentroidResource* _centroidResource;
    mercury::IndexMeta _indexMeta;
    mercury::IndexMeta _fragmentIndexMeta;
    uint16_t* _queryCodeFeature;
    CentroidCandidate _centroids;
    bool _withCodeFeature;

protected:
    distance_t* _distanceArray;
    size_t _fragmentNum;
    size_t _centroidNum;
    size_t _elemSize;
};

}; // namespace mercury

#endif //__MERCURY_INDEX_QUERY_DISTANCE_MATRIX_H
