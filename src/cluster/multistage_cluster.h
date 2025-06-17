/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     multistage_cluster.h
 *   \author   Hechong.xyf
 *   \date     Mar 2018
 *   \version  1.0.0
 *   \brief    Interface of Multistage Cluster
 */

#ifndef __MERCURY_MULTISTAGE_CLUSTER_H__
#define __MERCURY_MULTISTAGE_CLUSTER_H__

#include "cluster_centroid.h"
#include "reservoir_sample.h"
#include "framework/vector_holder.h"

namespace mercury {

/*! Multistage Cluster
 */
class MultistageCluster
{
public:
    typedef mercury::VectorHolder::FeatureType FeatureType;

    //! Constructor
    MultistageCluster(void)
        : _features(nullptr), _feature_type(static_cast<FeatureType>(0)),
          _feature_size(0), _features_count(0), _sample_features(), _centroids()
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

    //! Mount features of cluster (simpling)
    void mount(const ClusterFeatureAny *feats, size_t count,
               FeatureType feat_type, size_t feat_size, size_t sample_count)
    {
        _feature_type = feat_type;
        _feature_size = feat_size;

        if (count > sample_count) {
            ReservoirSample(feats, count, sample_count, &_sample_features);
            _features = _sample_features.data();
            _features_count = _sample_features.size();
        } else {
            _features = feats;
            _features_count = count;
        }
    }

    //! Retrieve centroids of cluster
    std::vector<MultiClusterCentroid> &getCentroids(void)
    {
        return _centroids;
    }

    //! Retrieve centroids of cluster
    const std::vector<MultiClusterCentroid> &getCentroids(void) const
    {
        return _centroids;
    }

    //! Retrieve centroids' cost of cluster
    double getCentroidsCost(void) const
    {
        double accum = 0.0;
        size_t total = 0u;
        for (const auto &it : _centroids) {
            accum += it.score();
            total += it.follows();
        }
        return (accum / total);
    }

    //! Cluster
    template <typename T, typename... TArgs>
    bool cluster(mercury::ThreadPool &pool, T &first, TArgs &... others)
    {
        if (!this->isValid()) {
            return false;
        }
        first.mount(_features, _features_count, _feature_type, _feature_size);
        return this->multiCluster(pool, _centroids, first, others...);
    }

    //! Classify
    template <typename T, typename... TArgs>
    bool classify(mercury::ThreadPool &pool, T &first, TArgs &... others)
    {
        if (!this->isValid()) {
            return false;
        }
        first.mount(_features, _features_count, _feature_type, _feature_size);
        return this->multiClassify(pool, _centroids, first, others...);
    }

protected:
    //! Multi Cluster (Empty)
    bool multiCluster(mercury::ThreadPool &,
                      std::vector<MultiClusterCentroid> &centroids)
    {
        centroids.clear();
        return true;
    }

    //! Multi Cluster (Only one)
    template <typename T>
    bool multiCluster(mercury::ThreadPool &pool,
                      std::vector<MultiClusterCentroid> &centroids, T &first)
    {
        if (!first.cluster(pool)) {
            return false;
        }

        // Update centroids
        centroids.clear();
        for (auto &it : first.getCentroids()) {
            centroids.emplace_back(std::move(it));
        }
        return true;
    }

    //! Multi Cluster
    template <typename T0, typename T1, typename... TArgs>
    bool multiCluster(mercury::ThreadPool &pool,
                      std::vector<MultiClusterCentroid> &centroids, T0 &first,
                      T1 &second, TArgs &... others)
    {
        if (!first.cluster(pool)) {
            return false;
        }

        if (!first.classify(pool)) {
            return false;
        }

        // Update centroids
        centroids.clear();
        auto &cluster_centroids = first.getCentroids();
        auto &cluster_features = first.getClusterFeatures();
        size_t features_total = 0;
        for (size_t i = 0; i < cluster_centroids.size(); ++i) {
            centroids.emplace_back(std::move(cluster_centroids[i]),
                                   std::move(cluster_features[i]));
            features_total += centroids[i].similars().size();
        }

        // The second clustering
        for (auto &it : centroids) {
            const auto &feats = it.similars();

            if (feats.empty()) {
                continue;
            }
            second.suggest((size_t)std::ceil((double)first.getClusterCount() *
                                             (double)second.getClusterCount() *
                                             (double)feats.size() /
                                             (double)features_total));
            second.mount(feats.data(), feats.size(), _feature_type,
                         _feature_size);
            if (!this->multiCluster(pool, it.subitems(), second, others...)) {
                return false;
            }
        }
        return true;
    }

    //! Multi Classify (Empty)
    bool multiClassify(mercury::ThreadPool &,
                       std::vector<MultiClusterCentroid> &)
    {
        return true;
    }

    //! Multi Classify (Only one)
    template <typename T>
    bool multiClassify(mercury::ThreadPool &pool,
                       std::vector<MultiClusterCentroid> &centroids, T &first)
    {
        std::vector<ClusterCentroid> input_centroids;

        for (auto &it : centroids) {
            input_centroids.emplace_back(std::move(it));
        }

        if (!first.classify(pool, std::move(input_centroids))) {
            return false;
        }

        // Update centroids
        auto &cluster_centroids = first.getCentroids();
        auto &cluster_features = first.getClusterFeatures();
        for (size_t i = 0; i < cluster_centroids.size(); ++i) {
            centroids[i].set(std::move(cluster_centroids[i]),
                             std::move(cluster_features[i]));
        }
        return true;
    }

    //! Multi Classify
    template <typename T0, typename T1, typename... TArgs>
    bool multiClassify(mercury::ThreadPool &pool,
                       std::vector<MultiClusterCentroid> &centroids, T0 &first,
                       T1 &second, TArgs &... others)
    {
        std::vector<ClusterCentroid> input_centroids;

        for (auto &it : centroids) {
            input_centroids.emplace_back(std::move(it));
        }

        if (!first.classify(pool, std::move(input_centroids))) {
            return false;
        }

        // Update centroids
        auto &cluster_centroids = first.getCentroids();
        auto &cluster_features = first.getClusterFeatures();
        for (size_t i = 0; i < cluster_centroids.size(); ++i) {
            centroids[i].set(std::move(cluster_centroids[i]),
                             std::move(cluster_features[i]));
        }

        // The second classifying
        for (auto &it : centroids) {
            const auto &feats = it.similars();

            if (feats.empty()) {
                continue;
            }
            second.mount(feats.data(), feats.size(), _feature_type,
                         _feature_size);
            if (!this->multiClassify(pool, it.subitems(), second, others...)) {
                return false;
            }
        }
        return true;
    }

private:
    const ClusterFeatureAny *_features;
    FeatureType _feature_type;
    size_t _feature_size;
    size_t _features_count;
    std::vector<ClusterFeatureAny> _sample_features;
    std::vector<MultiClusterCentroid> _centroids;
};

} // namespace mercury

#endif // __MERCURY_MULTISTAGE_CLUSTER_H__
