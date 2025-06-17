/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     kmedoids_cluster.cc
 *   \author   yunheng@xiaohongshu.com
 *   \date     Jan 2019
 *   \version  1.0.0
 *   \brief    Implementation of K-medoids Cluster
 */

#include "kmedoids_cluster.h"
#include "reservoir_sample.h"
#include "vector_mean.h"

#include "framework/index_logger.h"
#include "framework/utility/time_helper.h"
#include <algorithm>
#include <cfloat>
#include <random>
#include <set>

namespace mercury {

static inline std::shared_ptr<VectorMean>
NewVectorMean(KmedoidsCluster::FeatureType feat_type, size_t feat_size)
{
    switch (feat_type) {
    case IndexMeta::kTypeFloat:
        return std::make_shared<NumericalVectorMean<float>>(feat_size /
                                                            sizeof(float));

    case IndexMeta::kTypeDouble:
        return std::make_shared<NumericalVectorMean<double>>(feat_size /
                                                             sizeof(double));

    case IndexMeta::kTypeInt8:
        return std::make_shared<NumericalVectorMean<int8_t>>(feat_size /
                                                             sizeof(int8_t));

    case IndexMeta::kTypeInt16:
        return std::make_shared<NumericalVectorMean<int16_t>>(feat_size /
                                                              sizeof(int16_t));

    default:
        break;
    }
    // As binary default
    return std::make_shared<BinaryVectorMean>(feat_size * 8);
}

static inline std::shared_ptr<VectorMean>
NewVectorHarmonicMean(KmedoidsCluster::FeatureType feat_type, size_t feat_size)
{
    switch (feat_type) {
    case IndexMeta::kTypeFloat:
        return std::make_shared<NumericalVectorHarmonicMean<float>>(
            feat_size / sizeof(float));

    case IndexMeta::kTypeDouble:
        return std::make_shared<NumericalVectorHarmonicMean<double>>(
            feat_size / sizeof(double));

    case IndexMeta::kTypeInt8:
        return std::make_shared<NumericalVectorHarmonicMean<int8_t>>(
            feat_size / sizeof(int8_t));

    case IndexMeta::kTypeInt16:
        return std::make_shared<NumericalVectorHarmonicMean<int16_t>>(
            feat_size / sizeof(int16_t));

    default:
        break;
    }
    // As binary default
    return std::make_shared<BinaryVectorMean>(feat_size * 8);
}

static inline std::shared_ptr<VectorMean>
NewVectorGeometricMean(KmedoidsCluster::FeatureType feat_type, size_t feat_size)
{
    switch (feat_type) {
    case IndexMeta::kTypeFloat:
        return std::make_shared<NumericalVectorGeometricMean<float>>(
            feat_size / sizeof(float));

    case IndexMeta::kTypeDouble:
        return std::make_shared<NumericalVectorGeometricMean<double>>(
            feat_size / sizeof(double));

    case IndexMeta::kTypeInt8:
        return std::make_shared<NumericalVectorGeometricMean<int8_t>>(
            feat_size / sizeof(int8_t));

    case IndexMeta::kTypeInt16:
        return std::make_shared<NumericalVectorGeometricMean<int16_t>>(
            feat_size / sizeof(int16_t));

    default:
        break;
    }
    // As binary default
    return std::make_shared<BinaryVectorMean>(feat_size * 8);
}

void KmedoidsCluster::updateScoringInThread(ClusterFeatureAny entry,
                                            size_t index_begin,
                                            size_t index_end,
                                            double *score_table, double *total)
{
    double accum = *total;

    for (size_t i = index_begin; i != index_end; ++i) {
        ClusterFeatureAny feat = _features[i];

        if (entry != feat) {
            float score = _distance(entry, feat, _feature_size);
            accum += score;
            score_table[i] += score;
        }
    }
    *total = accum;
}

void KmedoidsCluster::updateClusterInThread(size_t index_begin,
                                            size_t index_end,
                                            size_t thread_index)
{
    std::vector<std::vector<ClusterFeature>> &cluster_features =
        _thread_cluster_features[thread_index];

    for (size_t i = index_begin; i != index_end; ++i) {
        ClusterFeatureAny feat = _features[i];
        size_t sel_column = std::numeric_limits<size_t>::max();
        float sel_score = std::numeric_limits<float>::max();

        for (size_t j = 0; j < _cluster_medoids.size(); ++j) {
            float score =
                _distance(_cluster_medoids[j].feature(), feat, _feature_size);
            if (std::isnan(score)) continue;
            if (score < sel_score) {
                sel_score = score;
                sel_column = j;
            }
        }
        if (sel_column < cluster_features.size()) {
            cluster_features[sel_column].emplace_back(sel_score, feat);
        }
    }
}

void KmedoidsCluster::updateCentroidAllInThread(size_t column)
{
    ClusterCentroid *centroid = &_cluster_medoids[column];
    size_t cluster_size = 0;
    double medoid_cost = 0.0;
    ClusterFeatureAny medoid_feature = centroid->feature();

    // Create Accumulator
    std::shared_ptr<VectorMean> accum;
    if (_mean_method == GEOMETRIC_MEAN) {
        accum = NewVectorGeometricMean(_feature_type, _feature_size);
    } else if (_mean_method == HARMONIC_MEAN) {
        accum = NewVectorHarmonicMean(_feature_type, _feature_size);
    } else {
        accum = NewVectorMean(_feature_type, _feature_size);
    }

    // Prepare data into heap
    std::vector<ClusterFeatureAny> heap;
    for (const auto &cluster_features : _thread_cluster_features) {
        const auto &cluster_feature = cluster_features[column];

        cluster_size += cluster_feature.size();
        for (const auto &it : cluster_feature) {
            heap.push_back(it.feature());
            medoid_cost += it.score();
            accum->plus(it.feature(), _feature_size);
        }
    }

    // Add the means into heap
    std::string means(_feature_size, 0);
    accum->mean(const_cast<char *>(means.data()), _feature_size);
    heap.push_back(means.data());

    // Prepare distance matrix
    std::vector<std::vector<float>> matrix(heap.size());
    for (auto &it : matrix) {
        it.clear();
        it.resize(heap.size(), 0.0f);
    }

    // Calculate scores in distance matrix
    for (size_t i = 0; i < heap.size(); ++i) {
        for (size_t j = 0; j < i; ++j) {
            float score = _distance(heap[i], heap[j], _feature_size);
            matrix[i][j] = score;
            matrix[j][i] = score;
        }
    }

    double selected_cost = 0.0;
    size_t selected_index = 0;
    for (auto score : matrix.front()) {
        selected_cost += score;
    }

    for (size_t i = 1; i < matrix.size(); ++i) {
        double cost = 0.0;
        for (auto score : matrix[i]) {
            cost += score;
        }

        if (cost < selected_cost) {
            selected_cost = cost;
            selected_index = i;
        }
    }

    if (selected_cost < medoid_cost) {
        // Save new centroid
        medoid_feature = heap[selected_index];
        medoid_cost = selected_cost;
    }

    // Update centroid
    centroid->set(medoid_feature, _feature_size, medoid_cost, cluster_size);
}

void KmedoidsCluster::updateCentroidInThread(size_t column)
{
    ClusterCentroid *centroid = &_cluster_medoids[column];
    size_t cluster_size = 0;
    double medoid_cost = 0.0;
    ClusterFeatureAny medoid_feature = centroid->feature();

    // Create Accumulator
    std::shared_ptr<VectorMean> accum;
    if (_mean_method == GEOMETRIC_MEAN) {
        accum = NewVectorGeometricMean(_feature_type, _feature_size);
    } else if (_mean_method == HARMONIC_MEAN) {
        accum = NewVectorHarmonicMean(_feature_type, _feature_size);
    } else {
        accum = NewVectorMean(_feature_type, _feature_size);
    }

    // Compute the cost and score of centroid
    for (const auto &cluster_features : _thread_cluster_features) {
        const auto &cluster_feature = cluster_features[column];

        cluster_size += cluster_feature.size();
        for (const auto &it : cluster_feature) {
            medoid_cost += it.score();
            accum->plus(it.feature(), _feature_size);
        }
    }

    // Prepare data into heap
    size_t bench_count =
        std::max((size_t)std::ceil(cluster_size * _bench_ratio), _min_benches);
    size_t bench_index = 0;

    std::random_device rd;
    std::mt19937 mt(rd());

    std::vector<ClusterFeatureAny> heap;
    for (const auto &cluster_features : _thread_cluster_features) {
        for (const auto &it : cluster_features[column]) {
            if (bench_index >= bench_count) {
                std::uniform_int_distribution<size_t> dist(0, bench_index);
                size_t j = dist(mt);

                if (j < bench_count) {
                    heap[j] = it.feature();
                }
            } else {
                heap.push_back(it.feature());
            }
            ++bench_index;
        }
    }

    // Add the means into heap
    std::string means(_feature_size, 0);
    accum->mean(const_cast<char *>(means.data()), _feature_size);
    heap.push_back(means.data());

    // Test every candidate in heap
    for (const auto &feat : heap) {
        double cost = this->getBenchCost(column, feat);
        if (!(cost < medoid_cost)) {
            continue;
        }

        // Save new centroid
        medoid_feature = feat;
        medoid_cost = cost;
    }

    // Update centroid
    centroid->set(medoid_feature, _feature_size, medoid_cost, cluster_size);
}

void KmedoidsCluster::updateCentroidOnlyMeanInThread(size_t column)
{
    size_t cluster_size = 0;
    double medoid_cost = 0.0;

    // Create Accumulator
    std::shared_ptr<VectorMean> accum;
    if (_mean_method == GEOMETRIC_MEAN) {
        accum = NewVectorGeometricMean(_feature_type, _feature_size);
    } else if (_mean_method == HARMONIC_MEAN) {
        accum = NewVectorHarmonicMean(_feature_type, _feature_size);
    } else {
        accum = NewVectorMean(_feature_type, _feature_size);
    }

    // Compute the cost and score of centroid
    for (const auto &cluster_features : _thread_cluster_features) {
        const auto &cluster_feature = cluster_features[column];
        cluster_size += cluster_feature.size();
        for (const auto &it : cluster_feature) {
            medoid_cost += it.score();
            accum->plus(it.feature(), _feature_size);
        }
    }

    // Update centroid
    ClusterCentroid *centroid = &_cluster_medoids[column];
    accum->mean(const_cast<void *>(centroid->feature()), _feature_size);
    centroid->set(medoid_cost, cluster_size);
}

void KmedoidsCluster::updateClusterFeaturesInThread(size_t column)
{
    std::vector<ClusterFeatureAny> *medoid_features =
        &_cluster_features[column];
    size_t cluster_size = 0;
    double medoid_cost = 0.0;

    // Size of cluster
    for (const auto &cluster_features : _thread_cluster_features) {
        cluster_size += cluster_features[column].size();
    }

    // Merge all features in cluster
    medoid_features->clear();
    medoid_features->reserve(cluster_size);
    for (const auto &cluster_features : _thread_cluster_features) {
        for (const auto &it : cluster_features[column]) {
            medoid_cost += it.score();
            medoid_features->push_back(it.feature());
        }
    }

    // Update centroid
    _cluster_medoids[column].set(medoid_cost, cluster_size);
}

void KmedoidsCluster::updateLabelsInThread(size_t index_begin, size_t index_end)
{
    for (size_t i = index_begin; i != index_end; ++i) {
        ClusterFeatureAny feat = _features[i];
        size_t sel_column = 0;
        float sel_score =
            _distance(_cluster_medoids[0].feature(), feat, _feature_size);

        for (size_t j = 1; j < _cluster_medoids.size(); ++j) {
            float score =
                _distance(_cluster_medoids[j].feature(), feat, _feature_size);
            if (score < sel_score) {
                sel_score = score;
                sel_column = j;
            }
        }
        _feature_labels[i] = sel_column;
    }
}

bool KmedoidsCluster::initCentroids(void)
{
    if (!this->isValid()) {
        return false;
    }

    // Selected centroids
    std::set<size_t> selected_points;

    // Random numbers' device
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dist(0, _features_count - 1);

    // Sample count
    size_t sample_count = std::min((_enable_suggest && _suggest_cluster_count)
                                       ? _suggest_cluster_count
                                       : _cluster_count,
                                   _features_count);

    // Selecting centroids
    while (selected_points.size() < sample_count) {
        selected_points.insert(dist(gen));
    }

    // Save centroids
    _cluster_medoids.clear();
    for (auto i : selected_points) {
        _cluster_medoids.emplace_back(_features[i], _feature_size);
    }
    return true;
}

bool KmedoidsCluster::initCentroids(mercury::ThreadPool &pool, size_t points)
{
    if (!this->isValid()) {
        return false;
    }

    // Score table
    std::vector<double> score_table;
    size_t entry_index = this->getFirstCentroid(pool, &score_table, points);
    ClusterFeatureAny entry_point = _features[entry_index];

    // Selected centroids
    std::set<size_t> selected_points;
    selected_points.insert(entry_index);

    // Random numbers' device
    std::random_device rd;
    std::mt19937 gen(rd());

    size_t shard_count = pool.count() * 2;
    size_t fregment_count = (_features_count + shard_count - 1) / shard_count;
    size_t sample_count = std::min((_enable_suggest && _suggest_cluster_count)
                                       ? _suggest_cluster_count
                                       : _cluster_count,
                                   _features_count);

    // Accums of every thread
    std::vector<double> thread_accums;
    thread_accums.resize(shard_count, 0.0);

    // Selecting centroids
    while (selected_points.size() < sample_count) {

        for (size_t i = 0, index = 0;
             (i != shard_count) && (index < _features_count); ++i) {

            size_t next_index = index + fregment_count;
            if (next_index > _features_count) {
                next_index = _features_count;
            }

            // Process in work thread
            pool.enqueue(mercury::Closure::New(
                             this, &KmedoidsCluster::updateScoringInThread,
                             entry_point, index, next_index, score_table.data(),
                             &thread_accums[i]),
                         true);

            // Next index
            index = next_index;
        }
        pool.waitFinish();

        // Compute the sum
        double accums =
            std::accumulate(thread_accums.begin(), thread_accums.end(), 0.0);
        std::uniform_real_distribution<double> real_dist(0.0, accums);

        bool try_again = false;
        do {
            double gen_score = real_dist(gen);

            for (size_t i = 0; i < _features_count; ++i) {
                gen_score -= score_table[i];

                if (gen_score <= 0) {
                    entry_point = _features[i];
                    try_again = !selected_points.insert(i).second;
                    break;
                }
            }
        } while (try_again);
    }

    // Save centroids
    _cluster_medoids.clear();
    for (auto i : selected_points) {
        _cluster_medoids.emplace_back(_features[i], _feature_size);
    }
    return true;
}

bool KmedoidsCluster::initCentroids(
    const std::vector<ClusterCentroid> &centroids)
{
    if (!this->isValid()) {
        return false;
    }

    if (centroids.empty()) {
        return false;
    }

    // Save centroids
    _cluster_medoids = centroids;
    return true;
}

bool KmedoidsCluster::initCentroids(std::vector<ClusterCentroid> &&centroids)
{
    if (!this->isValid()) {
        return false;
    }

    if (centroids.empty()) {
        return false;
    }

    // Save centroids
    _cluster_medoids = std::move(centroids);
    return true;
}

size_t KmedoidsCluster::getFirstCentroid(mercury::ThreadPool &pool,
                                         std::vector<double> *scores,
                                         size_t points)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dist(0, _features_count - 1);

    // Calculate count of sample
    size_t sample_count =
        std::min(std::max(points, (size_t)1), _features_count);

    // Selected points
    std::set<size_t> selected_points;
    while (selected_points.size() < sample_count) {
        selected_points.insert(dist(gen));
    }

    size_t shard_count = pool.count() * 2;
    size_t fregment_count = (_features_count + shard_count - 1) / shard_count;

    size_t entry_index = -1;
    double entry_accums = std::numeric_limits<double>::max();

    // Accums of every thread
    std::vector<double> thread_accums;
    std::vector<double> score_table;

    for (auto selected_index : selected_points) {
        ClusterFeatureAny entry_point = _features[selected_index];

        thread_accums.clear();
        thread_accums.resize(shard_count, 0.0);

        score_table.clear();
        score_table.resize(_features_count, 0.0);

        for (size_t i = 0, index = 0;
             (i != shard_count) && (index < _features_count); ++i) {

            size_t next_index = index + fregment_count;
            if (next_index > _features_count) {
                next_index = _features_count;
            }

            // Process in work thread
            pool.enqueue(mercury::Closure::New(
                             this, &KmedoidsCluster::updateScoringInThread,
                             entry_point, index, next_index, score_table.data(),
                             &thread_accums[i]),
                         true);

            // Next index
            index = next_index;
        }
        pool.waitFinish();

        // Compute the sum
        double accums =
            std::accumulate(thread_accums.begin(), thread_accums.end(), 0.0);
        if (accums < entry_accums) {
            entry_accums = accums;
            entry_index = selected_index;
            scores->swap(score_table);
        }
    }
    return entry_index;
}

void KmedoidsCluster::updateClusters(mercury::ThreadPool &pool)
{
    size_t shard_count = pool.count() * 2;
    size_t fregment_count = (_features_count + shard_count - 1) / shard_count;

    // Initilize containers
    _thread_cluster_features.resize(shard_count);

    for (auto &cluster_features : _thread_cluster_features) {
        cluster_features.resize(_cluster_medoids.size());

        // Clear output buffer first
        for (auto &it : cluster_features) {
            it.clear();
        }
    }

    for (size_t i = 0, index = 0;
         (i != shard_count) && (index < _features_count); ++i) {

        size_t next_index = index + fregment_count;
        if (next_index > _features_count) {
            next_index = _features_count;
        }

        // Process in work thread
        pool.enqueue(
            mercury::Closure::New(this, &KmedoidsCluster::updateClusterInThread,
                                  index, next_index, i),
            true);

        // Next index
        index = next_index;
    }
    pool.waitFinish();
}

void KmedoidsCluster::updateCentroids(mercury::ThreadPool &pool)
{
    if (_only_means) {
        for (size_t i = 0; i < _cluster_medoids.size(); ++i) {
            pool.enqueue(
                mercury::Closure::New(
                    this, &KmedoidsCluster::updateCentroidOnlyMeanInThread, i),
                true);
        }
    } else if (_bench_ratio >= 0.999) {
        for (size_t i = 0; i < _cluster_medoids.size(); ++i) {
            pool.enqueue(
                mercury::Closure::New(
                    this, &KmedoidsCluster::updateCentroidAllInThread, i),
                true);
        }
    } else {
        for (size_t i = 0; i < _cluster_medoids.size(); ++i) {
            pool.enqueue(mercury::Closure::New(
                             this, &KmedoidsCluster::updateCentroidInThread, i),
                         true);
        }
    }
    pool.waitFinish();
}

void KmedoidsCluster::updateClustersFeatures(mercury::ThreadPool &pool)
{
    // Initilize containers
    _cluster_features.resize(_cluster_medoids.size());

    for (size_t i = 0; i < _cluster_medoids.size(); ++i) {
        // Process in work thread
        pool.enqueue(
            mercury::Closure::New(
                this, &KmedoidsCluster::updateClusterFeaturesInThread, i),
            true);
    }
    pool.waitFinish();
}

void KmedoidsCluster::updateLabels(mercury::ThreadPool &pool)
{
    _feature_labels.resize(_features_count);

    size_t shard_count = pool.count() * 2;
    size_t fregment_count = (_features_count + shard_count - 1) / shard_count;

    for (size_t i = 0, index = 0;
         (i != shard_count) && (index < _features_count); ++i) {

        size_t next_index = index + fregment_count;
        if (next_index > _features_count) {
            next_index = _features_count;
        }

        // Process in work thread
        pool.enqueue(
            mercury::Closure::New(this, &KmedoidsCluster::updateLabelsInThread,
                                  index, next_index),
            true);

        // Next index
        index = next_index;
    }
    pool.waitFinish();
}

bool KmedoidsCluster::cluster(mercury::ThreadPool &pool, bool outfit)
{
    if (!this->isValid()) {
        return false;
    }
    if (outfit && !this->initCentroids()) {
        return false;
    }

    mercury::ElapsedTime stamp;

    // Do the first clustering
    this->updateClusters(pool);
    LOG_DEBUG("Update First %zu Clusters: %zu ms", _cluster_medoids.size(),
              stamp.update());

    this->updateCentroids(pool);
    LOG_DEBUG("Update First %zu Centroids: %zu ms", _cluster_medoids.size(),
              stamp.update());

    // Save cost of centroids
    double cost = this->getCentroidsCost();
    for (size_t i = 0; i < _max_iterations; ++i) {
        this->updateClusters(pool);
        this->updateCentroids(pool);

        // Calculate cost
        double new_cost = this->getCentroidsCost();
        double new_epsilon = std::abs(cost - new_cost);

        LOG_DEBUG("Update (%zu) Elapsed: %zu ms, Cost: %f -> %f, %f", i + 1,
                  stamp.update(), cost, new_cost, new_epsilon);

        if (new_epsilon < _epsilon || std::isnan(new_cost)) {
            std::cout << "breaking..." << std::endl;
            break;
        }
        cost = new_cost;
    }
    return true;
}

bool KmedoidsCluster::classify(mercury::ThreadPool &pool, bool outfit)
{
    if (!this->isValid()) {
        return false;
    }
    if (outfit && !this->initCentroids()) {
        return false;
    }
    if (_cluster_medoids.empty()) {
        return false;
    }

    this->updateClusters(pool);
    this->updateClustersFeatures(pool);
    return true;
}

bool KmedoidsCluster::label(mercury::ThreadPool &pool, bool outfit)
{
    if (!this->isValid()) {
        return false;
    }
    if (outfit && !this->initCentroids()) {
        return false;
    }
    if (_cluster_medoids.empty()) {
        return false;
    }

    this->updateLabels(pool);
    return true;
}

void KmedoidsCluster::mount(const ClusterFeatureAny *feats, size_t count,
                            FeatureType feat_type, size_t feat_size,
                            size_t sample_count)
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

} // namespace mercury
