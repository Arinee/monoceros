/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     coarse_vamana_index.h
 *   \author   shiyang
 *   \date     Nov 2023
 *   \version  1.0.0
 *   \brief    interface and impl of coarse_vamana_index
 */

#pragma once

#include "src/core/framework/index_logger.h"
#include "src/core/common/common.h"
#include "src/core/utils/vamana/in_mem_data_store.h"
#include "src/core/utils/vamana/in_mem_graph_store.h"
#include "src/core/utils/vamana/timer.h"
#include "src/core/algorithm/vamana/vamana_index_config.h"
#include "src/core/utils/vamana/scratch.h"
#include <omp.h>


MERCURY_NAMESPACE_BEGIN(core);
class CoarseVamanaIndex
{
public:
    CoarseVamanaIndex(uint16_t data_size, const IndexConfig &index_config, 
                        std::unique_ptr<InMemDataStore> data_store, std::unique_ptr<InMemGraphStore> graph_store, bool use_half);

    CoarseVamanaIndex(uint16_t data_size, const IndexConfig &index_config, IndexDistance::Methods method, bool use_half);

    ~CoarseVamanaIndex();

    void build_with_data_populated();

    void build(const char *filename, const size_t num_points_to_load);

    // save the graph index on a file as an adjacency list. For each point,
    // first store the number of neighbors, and then the neighbor list (each as
    // 4 byte uint32_t)
    size_t save_graph(std::string graph_file);

    size_t save_data(std::string data_file);

    void save(const char *filename);

    void dump(IndexPackage &index_package);

    void get_data_vec(const uint32_t i, void *dest) const;

    void set_data_vec(const uint32_t loc, const void *const vector);

    size_t load_data(std::string filename);

    size_t load_graph(std::string filename, size_t expected_num_points);

    void load(const char *filename, uint32_t num_threads, uint32_t search_l);

    std::pair<uint32_t, uint32_t> search(const void *query, const size_t K, const uint32_t L,
                                         uint64_t *indices, float *distance);

    size_t get_active_num();

private:
    uint16_t _data_size;
    size_t _dim;
    size_t _nd = 0;         // number of active points i.e. existing in the graph
    std::unordered_set<uint32_t> _id_set; // set of active point ids 
    size_t _max_points = 0; // total number of points in given data set

    // _num_frozen_pts is the number of points which are used as initial
    // candidates when iterating to closest point(s). These are not visible
    // externally and won't be returned by search. At least 1 frozen point is
    // needed for a dynamic index. The frozen points have consecutive locations.
    // See also _start below.
    size_t _num_frozen_pts = 0;
    size_t _frozen_pts_used = 0;
    size_t _node_size;
    size_t _data_len;
    size_t _neighbor_len;

    //  Start point of the search. When _num_frozen_pts is greater than zero,
    //  this is the location of the first frozen point. Otherwise, this is a
    //  location of one of the points in index.
    uint32_t _start = 0;

    bool _has_built = false;
    bool _saturate_graph = false;
    bool _dynamic_index = false;
    bool _enable_tags = false;
    bool _deletes_enabled = false;

    // Indexing parameters
    uint32_t _indexingQueueSize;
    uint32_t _indexingRange;
    uint32_t _indexingMaxC;
    float _indexingAlpha;
    uint32_t _indexingThreads;

    // Per node lock, cardinality=_max_points + _num_frozen_points
    std::vector<std::mutex> _locks;

    // Query scratch data structures
    ConcurrentQueue<InMemQueryScratch *> _query_scratch;

public:

    bool _use_half;

    // Data
    std::unique_ptr<InMemDataStore> _data_store;

    // Graph related data structures
    std::unique_ptr<InMemGraphStore> _graph_store;

private:
    void initialize_query_scratch(uint32_t num_threads, uint32_t search_l, uint32_t indexing_l, uint32_t r, uint32_t maxc, size_t dim);

    uint32_t calculate_entry_point();

    std::vector<uint32_t> get_init_ids();

    std::pair<uint32_t, uint32_t> iterate_to_fixed_point(const void *query, const uint32_t Lsize, const std::vector<uint32_t> &init_ids, 
                                                            InMemQueryScratch *scratch);

    void occlude_list(const uint32_t location, std::vector<Neighbor> &pool, const float alpha,
                        const uint32_t degree, const uint32_t maxc, std::vector<uint32_t> &result, InMemQueryScratch *scratch);

    void prune_neighbors(const uint32_t location, std::vector<Neighbor> &pool, const uint32_t range,
                            const uint32_t max_candidate_size, const float alpha,
                            std::vector<uint32_t> &pruned_list, InMemQueryScratch *scratch);

    void prune_neighbors(const uint32_t location, std::vector<Neighbor> &pool,
                            std::vector<uint32_t> &pruned_list, InMemQueryScratch *scratch);

    void search_for_point_and_prune(int location, uint32_t Lindex, std::vector<uint32_t> &pruned_list, InMemQueryScratch *scratch);

    void inter_insert(uint32_t n, std::vector<uint32_t> &pruned_list, const uint32_t range, InMemQueryScratch *scratch);

    void inter_insert(uint32_t n, std::vector<uint32_t> &pruned_list, InMemQueryScratch *scratch);

    void link();

    void reposition_points(uint32_t old_location_start, uint32_t new_location_start, uint32_t num_locations);

    void resize(size_t new_max_points);

};

MERCURY_NAMESPACE_END(core);