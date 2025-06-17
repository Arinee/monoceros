/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     cluster_feature.h
 *   \author   yunheng@xiaohongshu.com
 *   \date     Jan 2019
 *   \version  1.0.0
 *   \brief    Implementation of Cluster Feature
 */

#ifndef __MERCURY_CLUSTER_CLUSTER_FEATURE_H__
#define __MERCURY_CLUSTER_CLUSTER_FEATURE_H__

namespace mercury {

//! Cluster Feature Any
typedef const void *ClusterFeatureAny;

/*! Cluster Feature
 */
struct ClusterFeature
{
public:
    //! Constructor
    ClusterFeature(void) : _score(0.0), _feature(nullptr) {}

    //! Constructor
    ClusterFeature(float val, ClusterFeatureAny feat)
        : _score(val), _feature(feat)
    {
    }

    //! Constructor
    ClusterFeature(const ClusterFeature &rhs)
        : _score(rhs._score), _feature(rhs._feature)
    {
    }

    //! Assignment
    ClusterFeature &operator=(const ClusterFeature &rhs)
    {
        _score = rhs._score;
        _feature = rhs._feature;
        return *this;
    }

    //! Less than
    bool operator<(const ClusterFeature &rhs) const
    {
        return (this->_score < rhs._score);
    }

    //! Retrieve score of feature
    float score(void) const
    {
        return _score;
    }

    //! Retrieve raw pointer of feature
    ClusterFeatureAny feature(void) const
    {
        return _feature;
    }

private:
    //! Members
    float _score;
    ClusterFeatureAny _feature;
};

} // namespace mercury

#endif // __MERCURY_CLUSTER_CLUSTER_FEATURE_H__
