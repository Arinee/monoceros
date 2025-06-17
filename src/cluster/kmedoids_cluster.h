/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     kmedoids_cluster.h
 *   \author   Hechong.xyf
 *   \date     Mar 2018
 *   \version  1.0.0
 *   \brief    Interface of K-medoids Cluster
 */

#ifndef __MERCURY_CLUSTER_KMEDOIDS_CLUSTER_H__
#define __MERCURY_CLUSTER_KMEDOIDS_CLUSTER_H__

#include "framework/index_distance.h"
#include "framework/index_meta.h"
#include "framework/utility/thread_pool.h"
#include "cluster_centroid.h"
#include <numeric>

namespace mercury {

/*! K-medoids Cluster
 */
class KmedoidsCluster
{
public:
    typedef IndexDistance::Measure MeasureType;
    typedef IndexMeta::FeatureTypes FeatureType;

    /*! Mean Methods
     */
    enum MeanMethods
    {
        ARITHMETIC_MEAN = 0,
        GEOMETRIC_MEAN = 1,
        HARMONIC_MEAN = 2,
    };
    
    //! Constructor
    KmedoidsCluster(void)
        : _features(nullptr), _feature_type(static_cast<FeatureType>(0)),
          _feature_size(0), _features_count(0), _min_benches(3u),
          _bench_ratio(0.25), _epsilon(std::numeric_limits<float>::epsilon()),
          _max_iterations(20), _cluster_count(0), _suggest_cluster_count(0),
          _only_means(false), _enable_suggest(false),
          _mean_method(ARITHMETIC_MEAN), _distance(), _sample_features(),
          _cluster_medoids(), _cluster_features(), _thread_cluster_features(),
          _feature_labels()
    {
    }

    //! Test if it is valid
    bool isValid(void) const
    {
        if (!_cluster_count || !_feature_size || !_features_count ||
            !_features || !_distance) {
            return false;
        }
        return true;
    }

    //! Retrieve count of cluster
    size_t getClusterCount(void) const
    {
        return _cluster_count;
    }

    //! Suggest options of cluster
    void suggest(size_t k)
    {
        _suggest_cluster_count = k;
    }

    //! Set min of benches
    void setMinBenches(size_t limit)
    {
        _min_benches = std::max(limit, (size_t)3u);
    }

    //! Set bench ratio
    void setBenchRatio(double ratio)
    {
        _bench_ratio = ratio;
    }

    //! Set epsilon
    void setEpsilon(double epsilon)
    {
        _epsilon = epsilon;
    }

    //! Set max of iterations
    void setMaxIterations(size_t limit)
    {
        _max_iterations = limit;
    }

    //! Set count of cluster
    void setClusterCount(size_t k)
    {
        _cluster_count = k;
    }

    //! Set only means of cluster
    void setOnlyMeans(bool val)
    {
        _only_means = val;
    }

    //! Set suggest option of cluster
    void setEnableSuggest(bool val)
    {
        _enable_suggest = val;
    }

    //! Set mean method of cluster
    void setMeanMethod(MeanMethods val)
    {
        _mean_method = val;
    }

    //! Set distance measure of cluster
    void setDistance(const MeasureType &func)
    {
        _distance = func;
    }

    //! Initialize Centroids (default)
    bool initCentroids(void);

    //! Initialize Centroids (k-means++)
    bool initCentroids(mercury::ThreadPool &pool, size_t points);

    //! Initialize Centroids
    bool initCentroids(const std::vector<ClusterCentroid> &centroids);

    //! Initialize Centroids
    bool initCentroids(std::vector<ClusterCentroid> &&centroids);

    //! Cluster
    bool cluster(mercury::ThreadPool &pool, bool outfit);

    //! Classify
    bool classify(mercury::ThreadPool &pool, bool outfit);

    //! Label
    bool label(mercury::ThreadPool &pool, bool outfit);

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
               FeatureType feat_type, size_t feat_size, size_t sample_count);

    //! Clear Cluster
    void clear(void)
    {
        _cluster_medoids.clear();
        _cluster_features.clear();
        _thread_cluster_features.clear();
        _feature_labels.clear();
    }

    //! Retrieve centroids of cluster
    const std::vector<ClusterCentroid> &getCentroids(void) const
    {
        return _cluster_medoids;
    }

    //! Retrieve centroids of cluster
    std::vector<ClusterCentroid> &getCentroids(void)
    {
        return _cluster_medoids;
    }

    //! Retrieve centroids' cost of cluster
    double getCentroidsCost(void) const
    {
        double accum = 0.0;
        size_t total = 0u;
        for (const auto &it : _cluster_medoids) {
            accum += it.score();
            total += it.follows();
        }
        return (accum / total);
    }

    //! Retrieve features of every cluster
    const std::vector<std::vector<ClusterFeatureAny>> &
    getClusterFeatures(void) const
    {
        return _cluster_features;
    }

    //! Retrieve features of every cluster
    std::vector<std::vector<ClusterFeatureAny>> &getClusterFeatures(void)
    {
        return _cluster_features;
    }

    //! Retrieve labels of every feature
    const std::vector<size_t> &getLabels(void) const
    {
        return _feature_labels;
    }

    //! Retrieve labels of every feature
    std::vector<size_t> &getLabels(void)
    {
        return _feature_labels;
    }

    //! Cluster
    bool cluster(mercury::ThreadPool &pool)
    {
        return this->cluster(pool, true);
    }

    //! Cluster
    bool cluster(mercury::ThreadPool &pool,
                 const std::vector<ClusterCentroid> &centroids)
    {
        if (!this->initCentroids(centroids)) {
            return false;
        }
        return this->cluster(pool, false);
    }

    //! Cluster
    bool cluster(mercury::ThreadPool &pool,
                 std::vector<ClusterCentroid> &&centroids)
    {
        if (!this->initCentroids(std::move(centroids))) {
            return false;
        }
        return this->cluster(pool, false);
    }

    //! Classify
    bool classify(mercury::ThreadPool &pool)
    {
        return this->classify(pool, false);
    }

    //! Classify
    bool classify(mercury::ThreadPool &pool,
                  const std::vector<ClusterCentroid> &centroids)
    {
        if (!this->initCentroids(centroids)) {
            return false;
        }
        return this->classify(pool, false);
    }

    //! Classify
    bool classify(mercury::ThreadPool &pool,
                  std::vector<ClusterCentroid> &&centroids)
    {
        if (!this->initCentroids(std::move(centroids))) {
            return false;
        }
        return this->classify(pool, false);
    }

    //! Label
    bool label(mercury::ThreadPool &pool)
    {
        return this->label(pool, false);
    }

    //! Label
    bool label(mercury::ThreadPool &pool,
               const std::vector<ClusterCentroid> &centroids)
    {
        if (!this->initCentroids(centroids)) {
            return false;
        }
        return this->label(pool, false);
    }

    //! Label
    bool label(mercury::ThreadPool &pool,
               std::vector<ClusterCentroid> &&centroids)
    {
        if (!this->initCentroids(std::move(centroids))) {
            return false;
        }
        return this->label(pool, false);
    }

protected:
    //! Update Centroids
    void updateCentroids(mercury::ThreadPool &pool);

    //! Update Clusters' Features
    void updateClustersFeatures(mercury::ThreadPool &pool);

    //! Update Clusters
    void updateClusters(mercury::ThreadPool &pool);

    //! Update Labels
    void updateLabels(mercury::ThreadPool &pool);

    //! Get the First Centroid
    size_t getFirstCentroid(mercury::ThreadPool &pool,
                            std::vector<double> *scores, size_t points);

    //! Compute Bench Cost
    double getBenchCost(size_t column, ClusterFeatureAny feat) const
    {
        double accum = 0.0;

        for (const auto &cluster_features : _thread_cluster_features) {
            for (const auto &it : cluster_features[column]) {
                accum += _distance(it.feature(), feat, _feature_size);
            }
        }
        return accum;
    }

    //! Scoring in thread
    void updateScoringInThread(ClusterFeatureAny entry, size_t index_begin,
                               size_t index_end, double *score_table,
                               double *total);

    //! Update Cluster in thread
    void updateClusterInThread(size_t index_begin, size_t index_end,
                               size_t thread_index);

    //! Update Centroid in thread
    void updateCentroidAllInThread(size_t column);

    //! Update Centroid in thread
    void updateCentroidInThread(size_t column);

    //! Update Centroid Mean in thread
    void updateCentroidOnlyMeanInThread(size_t column);

    //! Update Cluster's Features in thread
    void updateClusterFeaturesInThread(size_t column);

    //! Update Labels in thread
    void updateLabelsInThread(size_t index_begin, size_t index_end);

private:
    const ClusterFeatureAny *_features;
    FeatureType _feature_type;
    size_t _feature_size;
    size_t _features_count;
    size_t _min_benches;
    double _bench_ratio;
    double _epsilon;
    size_t _max_iterations;
    size_t _cluster_count;
    size_t _suggest_cluster_count;
    bool _only_means;
    bool _enable_suggest;
    MeanMethods _mean_method;
    MeasureType _distance;
    std::vector<ClusterFeatureAny> _sample_features;
    std::vector<ClusterCentroid> _cluster_medoids;
    std::vector<std::vector<ClusterFeatureAny>> _cluster_features;
    std::vector<std::vector<std::vector<ClusterFeature>>>
        _thread_cluster_features;
    std::vector<size_t> _feature_labels;
};

} // namespace mercury

#endif // __MERCURY_CLUSTER_KMEDOIDS_CLUSTER_H__
