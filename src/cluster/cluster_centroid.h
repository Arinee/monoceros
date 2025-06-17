/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     cluster_centroid.h
 *   \author   yunheng@xiaohongshu.com
 *   \date     Jan 2019
 *   \version  1.0.0
 *   \brief    Implementation of Cluster Centroid
 */

#ifndef __MERCURY_CLUSTER_CLUSTER_CENTROID_H__
#define __MERCURY_CLUSTER_CLUSTER_CENTROID_H__

#include "cluster_feature.h"
#include "framework/index_framework.h"

namespace mercury {

/*! Cluster Centroid
 */
class ClusterCentroid
{
public:
    //! Constructor
    ClusterCentroid(void) : _centroid(), _score(0.0), _follows(0) {}

    //! Constructor
    ClusterCentroid(ClusterFeatureAny feat, size_t bytes)
        : _centroid(std::string(reinterpret_cast<const char *>(feat), bytes)),
          _score(0.0), _follows(0)
    {
    }

    //! Constructor
    ClusterCentroid(const ClusterCentroid &rhs)
        : _centroid(rhs._centroid), _score(rhs._score), _follows(rhs._follows)
    {
    }

    //! Constructor
    ClusterCentroid(ClusterCentroid &&rhs)
        : _centroid(std::move(rhs._centroid)), _score(rhs._score),
          _follows(rhs._follows)
    {
    }

    //! Assignment
    ClusterCentroid &operator=(const ClusterCentroid &rhs)
    {
        _centroid = rhs._centroid;
        _score = rhs._score;
        _follows = rhs._follows;
        return *this;
    }

    //! Assignment
    ClusterCentroid &operator=(ClusterCentroid &&rhs)
    {
        _centroid = std::move(rhs._centroid);
        _score = rhs._score;
        _follows = rhs._follows;
        return *this;
    }

    //! Less than
    bool operator<(const ClusterCentroid &rhs) const
    {
        return (this->_score < rhs._score);
    }

    //! Set the information of centroid
    void set(double val, size_t count)
    {
        _score = val;
        _follows = count;
    }

    //! Set the information of centroid
    void set(ClusterFeatureAny feat, size_t bytes, double val, size_t count)
    {
        _centroid.assign(
            std::string(reinterpret_cast<const char *>(feat), bytes));
        _score = val;
        _follows = count;
    }

    //! Retrieve feature pointer
    ClusterFeatureAny feature(void) const
    {
        return _centroid.data();
    }

    //! Retrieve size of centroid in bytes
    size_t size(void) const
    {
        return _centroid.size();
    }

    //! Retrieve score of centroid
    double score(void) const
    {
        return _score;
    }

    //! Retrieve follows of centroid
    size_t follows(void) const
    {
        return _follows;
    }

private:
    //! Members
    std::string _centroid;
    double _score;
    size_t _follows;
};

/*! Multi Cluster Centroid
 */
class MultiClusterCentroid : public ClusterCentroid
{
public:
    //! Constructor
    MultiClusterCentroid(void) : ClusterCentroid(), _subitems(), _similars() {}

    //! Constructor
    MultiClusterCentroid(const MultiClusterCentroid &rhs)
        : ClusterCentroid(rhs), _subitems(rhs._subitems),
          _similars(rhs._similars)
    {
    }

    //! Constructor
    MultiClusterCentroid(MultiClusterCentroid &&rhs)
        : ClusterCentroid(std::move(rhs)), _subitems(std::move(rhs._subitems)),
          _similars(std::move(rhs._similars))
    {
    }

    //! Constructor
    MultiClusterCentroid(const ClusterCentroid &rhs)
        : ClusterCentroid(rhs), _subitems(), _similars()
    {
    }

    //! Constructor
    MultiClusterCentroid(ClusterCentroid &&rhs)
        : ClusterCentroid(std::move(rhs)), _subitems(), _similars()
    {
    }

    //! Constructor
    MultiClusterCentroid(const ClusterCentroid &cent,
                         const std::vector<ClusterFeatureAny> &feats)
        : ClusterCentroid(cent), _subitems(), _similars(feats)
    {
    }

    //! Constructor
    MultiClusterCentroid(ClusterCentroid &&cent,
                         std::vector<ClusterFeatureAny> &&feats)
        : ClusterCentroid(std::move(cent)), _subitems(),
          _similars(std::move(feats))
    {
    }

    //! Constructor
    MultiClusterCentroid(const ClusterCentroid &cent,
                         std::vector<ClusterFeatureAny> &&feats)
        : ClusterCentroid(cent), _subitems(), _similars(std::move(feats))
    {
    }

    //! Constructor
    MultiClusterCentroid(ClusterCentroid &&cent,
                         const std::vector<ClusterFeatureAny> &feats)
        : ClusterCentroid(std::move(cent)), _subitems(), _similars(feats)
    {
    }

    //! Assignment
    MultiClusterCentroid &operator=(const MultiClusterCentroid &rhs)
    {
        *static_cast<ClusterCentroid *>(this) = rhs;
        _subitems = rhs._subitems;
        _similars = rhs._similars;
        return *this;
    }

    //! Assignment
    MultiClusterCentroid &operator=(MultiClusterCentroid &&rhs)
    {
        *static_cast<ClusterCentroid *>(this) = std::move(rhs);
        _subitems = std::move(rhs._subitems);
        _similars = std::move(rhs._similars);
        return *this;
    }

    //! Set the information
    void set(const ClusterCentroid &cent,
             const std::vector<ClusterFeatureAny> &feats)
    {
        *static_cast<ClusterCentroid *>(this) = cent;
        _similars = feats;
    }

    //! Set the information
    void set(ClusterCentroid &&cent, std::vector<ClusterFeatureAny> &&feats)
    {
        *static_cast<ClusterCentroid *>(this) = std::move(cent);
        _similars = std::move(feats);
    }

    //! Set the information
    void set(const ClusterCentroid &cent,
             std::vector<ClusterFeatureAny> &&feats)
    {
        *static_cast<ClusterCentroid *>(this) = cent;
        _similars = std::move(feats);
    }

    //! Set the information
    void set(ClusterCentroid &&cent,
             const std::vector<ClusterFeatureAny> &feats)
    {
        *static_cast<ClusterCentroid *>(this) = std::move(cent);
        _similars = feats;
    }

    //! Retrieve the sub centroids
    const std::vector<MultiClusterCentroid> &subitems(void) const
    {
        return _subitems;
    }

    //! Retrieve the sub centroids
    std::vector<MultiClusterCentroid> &subitems(void)
    {
        return _subitems;
    }

    //! Retrieve the similars
    const std::vector<ClusterFeatureAny> &similars(void) const
    {
        return _similars;
    }

    //! Retrieve the similars
    std::vector<ClusterFeatureAny> &similars(void)
    {
        return _similars;
    }

private:
    std::vector<MultiClusterCentroid> _subitems;
    std::vector<ClusterFeatureAny> _similars;
};

} // namespace mercury

#endif // __MERCURY_CLUSTER_CLUSTER_CENTROID_H__
