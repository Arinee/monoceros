#ifndef __MERCURY_INDEX_PQ_COMMON_H__
#define __MERCURY_INDEX_PQ_COMMON_H__

#include <vector>
#include <string>
#include <map>
#include "src/core/common/common.h"

MERCURY_NAMESPACE_BEGIN(core);

#define INVALID_CATEGORY (-1)

typedef float feature_t;
typedef float distance_t;
typedef uint8_t q_distance_t;
typedef float score_t;

typedef std::pair<docid_t, distance_t> DocIdToDistance;
typedef std::vector<DocIdToDistance> DocIdToDistanceVector;
typedef std::vector<DocIdToDistanceVector> DocIdToDistanceVectors;

typedef std::vector<docid_t> DocVector;
typedef std::vector<int32_t> IntVector;

typedef std::vector<uint8_t> UInt8Vector;
typedef std::vector<int16_t> Int16Vector;
typedef std::vector<uint16_t> UInt16Vector;
typedef std::vector<uint32_t> UInt32Vector;
typedef std::vector<uint64_t> UInt64Vector;

typedef std::vector<feature_t> FeatureVector;
typedef std::vector<distance_t> DistanceVector;
typedef std::vector<DistanceVector> DistanceVectors;

typedef std::map<std::string, std::string> StrToStrMap;

struct DistNode
{
    DistNode()
        : key(0L), dist(0.0f), offset(-1)
    {}; 
    DistNode(uint64_t k, distance_t d)
        : key(k), dist(d), offset(-1)
    {};
    DistNode(uint64_t k, distance_t d, int o)
        : key(k), dist(d), offset(o)
    {};
    bool operator< (const DistNode& right) const
    {
        return (dist < right.dist);
    }
    uint32_t key;
    distance_t dist;
    int offset;
};

struct DistGreater
{
    bool operator() (const DistNode& left,
            const DistNode& right)
    {
        return (left.dist > right.dist);
    }
};

struct DistLess
{
    bool operator() (const DistNode& left,
            const DistNode& right)
    {
        return (left.dist < right.dist);
    }
};



MERCURY_NAMESPACE_END(core);
#endif //__MERCURY_INDEX_PQ_COMMON_H__
