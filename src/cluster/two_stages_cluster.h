/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     two_stages_cluster.h
 *   \author   yunheng@xiaohongshu.com
 *   \date     Jan 2019
 *   \version  1.0.0
 *   \brief    Interface of Two Stages Cluster
 */

#ifndef __MERCURY_CLUSTER_TWO_STAGES_CLUSTER_H__
#define __MERCURY_CLUSTER_TWO_STAGES_CLUSTER_H__

#include "cluster_centroid.h"

namespace mercury {

/*! Two Stages Cluster
 */
class TwoStagesCluster
{
public:
    typedef IndexMeta::FeatureTypes FeatureType;

    //! Constructor
    TwoStagesCluster(void)
        : _features(nullptr), _feature_type(static_cast<FeatureType>(0)),
          _feature_size(0), _features_count(0), _cluster_medoid_cost_l1(0.0),
          _cluster_centroids_cost_l2(), _cluster_centroids_l1(),
          _cluster_centroids_l2()
    {
    }

    //! Test if it is valid
    bool isValid(void) const
    {
        if (!_feature_size || !_features_count || !_features) {
            return false;
        }
        return true;
    }

    //! Mount features of cluster
    void mount(const ClusterFeatureAny *feats, size_t count,
               FeatureType feat_type, size_t feat_size)
    {
        _features = feats;
        _features_count = count;
        _feature_type = feat_type;
        _feature_size = feat_size;
    }

    //! Retrieve centroids of cluster (level 1)
    const std::vector<ClusterCentroid> &getCentroidsLevel1(void) const
    {
        return _cluster_centroids_l1;
    }

    std::vector<ClusterCentroid> &getCentroidsLevel1(void)
    {
        return _cluster_centroids_l1;
    }

    //! Retrieve centroids' cost of cluster (level 1)
    double getCentroidsCostLevel1(void) const
    {
        return _cluster_medoid_cost_l1;
    }

    //! Retrieve centroids of cluster (level 2)
    const std::vector<std::vector<ClusterCentroid>> &
    getCentroidsLevel2(void) const
    {
        return _cluster_centroids_l2;
    }

    std::vector<std::vector<ClusterCentroid>> &getCentroidsLevel2(void)
    {
        return _cluster_centroids_l2;
    }

    //! Retrieve centroids' cost of cluster (level 2)
    const std::vector<double> &getCentroidsCostLevel2(void) const
    {
        return _cluster_centroids_cost_l2;
    }

    //! Retrieve features of every cluster
    const std::vector<std::vector<std::vector<ClusterFeatureAny>>> &
    getClusterFeatures(void) const
    {
        return _cluster_features;
    }

    //! Cluster
    template <typename T0, typename T1>
    bool cluster(mercury::ThreadPool &pool, T0 &first, T1 &second)
    {
        if (!this->isValid()) {
            return false;
        }
        _cluster_centroids_l2.clear();
        _cluster_centroids_cost_l2.clear();

        first.mount(_features, _features_count, _feature_type, _feature_size);
        if (!first.cluster(pool)) {
            return false;
        }

        if (!first.classify(pool)) {
            return false;
        }

        // Level 1 clustering
        _cluster_medoid_cost_l1 = first.getCentroidsCost();
        _cluster_centroids_l1 = std::move(first.getCentroids());

        std::vector<std::vector<ClusterFeatureAny>> cluster_features =
            std::move(first.getClusterFeatures());

        // Level 2 clustering
        for (auto &it : cluster_features) {
            std::vector<ClusterCentroid> centroids;
            double centroids_cost = 0.0;

            if (!it.empty()) {
                second.suggest((size_t)std::ceil(
                    (double)first.getClusterCount() *
                    (double)second.getClusterCount() * (double)it.size() /
                    (double)_features_count));

                second.mount(it.data(), it.size(), _feature_type,
                             _feature_size);
                if (!second.cluster(pool)) {
                    return false;
                }
                centroids_cost = second.getCentroidsCost();
                centroids = std::move(second.getCentroids());
            }
            _cluster_centroids_l2.push_back(std::move(centroids));
            _cluster_centroids_cost_l2.push_back(centroids_cost);
        }
        return true;
    }

    //! Classify
    template <typename T0, typename T1>
    bool classify(mercury::ThreadPool &pool, T0 &first, T1 &second)
    {
        if (!this->isValid()) {
            return false;
        }
        _cluster_features.clear();

        // Level 1 classifing
        first.clear();
        first.mount(_features, _features_count, _feature_type, _feature_size);
        if (!first.classify(pool, _cluster_centroids_l1)) {
            return false;
        }

        // Level 2 classifing
        std::vector<std::vector<ClusterFeatureAny>> &features_level1 =
            first.getClusterFeatures();

        _cluster_features.resize(features_level1.size());

        for (size_t i = 0; i < features_level1.size(); ++i) {
            std::vector<ClusterFeatureAny> &it = features_level1[i];

            if (!it.empty()) {
                // Level 2 classifing
                second.clear();
                second.mount(it.data(), it.size(), _feature_type,
                             _feature_size);
                if (!second.classify(pool, _cluster_centroids_l2[i])) {
                    return false;
                }
                _cluster_features[i] = std::move(second.getClusterFeatures());
            }
        }
        return true;
    }

private:
    const ClusterFeatureAny *_features;
    FeatureType _feature_type;
    size_t _feature_size;
    size_t _features_count;
    double _cluster_medoid_cost_l1;
    std::vector<double> _cluster_centroids_cost_l2;
    std::vector<ClusterCentroid> _cluster_centroids_l1;
    std::vector<std::vector<ClusterCentroid>> _cluster_centroids_l2;
    std::vector<std::vector<std::vector<ClusterFeatureAny>>> _cluster_features;
};

} // namespace mercury

#endif // __MERCURY_CLUSTER_TWO_STAGES_CLUSTER_H__
