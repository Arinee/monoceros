/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     coarse_vamana_index.cc
 *   \author   shiyang
 *   \date     Nov 2023
 *   \version  1.0.0
 *   \brief    interface and impl of coarse_vamana_index
 */

#include "coarse_vamana_index.h"

MERCURY_NAMESPACE_BEGIN(core);

CoarseVamanaIndex::CoarseVamanaIndex(uint16_t data_size, const IndexConfig &index_config, std::unique_ptr<InMemDataStore> data_store,
                            std::unique_ptr<InMemGraphStore> graph_store, bool use_half)
: _data_size(data_size), _dim(index_config.dimension), _max_points(index_config.max_points), 
    _num_frozen_pts(index_config.num_frozen_pts), _indexingMaxC(DEFAULT_MAXC), _query_scratch(nullptr), _use_half(use_half)
{
    const size_t total_internal_points = _max_points + _num_frozen_pts;
    _start = (uint32_t)_max_points;

    _data_store = std::move(data_store);
    _graph_store = std::move(graph_store);

    _locks = std::vector<std::mutex>(total_internal_points);
    if (index_config.index_write_params != nullptr)
    {
        _indexingQueueSize = index_config.index_write_params->search_list_size;
        _indexingRange = index_config.index_write_params->max_degree;
        _indexingMaxC = index_config.index_write_params->max_occlusion_size;
        _indexingAlpha = index_config.index_write_params->alpha;
        _indexingThreads = index_config.index_write_params->num_threads;
        _saturate_graph = index_config.index_write_params->saturate_graph;
    }
}

CoarseVamanaIndex::CoarseVamanaIndex(uint16_t data_size, const IndexConfig &index_config, IndexDistance::Methods method, bool use_half)
: _data_size(data_size), _dim(index_config.dimension), _max_points(index_config.max_points), 
    _num_frozen_pts(index_config.num_frozen_pts), _indexingMaxC(DEFAULT_MAXC), _query_scratch(nullptr), _use_half(use_half)
{
    const size_t total_internal_points = _max_points + _num_frozen_pts;
    _start = (uint32_t)_max_points;

    _data_store = std::make_unique<InMemDataStore>(index_config.max_points, index_config.dimension, _data_size);
    _data_store->setMethod(method);

    _graph_store = std::make_unique<InMemGraphStore>(index_config.max_points, index_config.max_reserve_degree);

    _locks = std::vector<std::mutex>(total_internal_points);
    if (index_config.index_write_params != nullptr)
    {
        _indexingQueueSize = index_config.index_write_params->search_list_size;
        _indexingRange = index_config.index_write_params->max_degree;
        _indexingMaxC = index_config.index_write_params->max_occlusion_size;
        _indexingAlpha = index_config.index_write_params->alpha;
        _indexingThreads = index_config.index_write_params->num_threads;
        _saturate_graph = index_config.index_write_params->saturate_graph;
    }
}

CoarseVamanaIndex::~CoarseVamanaIndex()
{
    for (auto &lock : _locks)
    {
        std::lock_guard<std::mutex> guard(lock);
    }

    if (!_query_scratch.empty())
    {
        ScratchStoreManager<InMemQueryScratch> manager(_query_scratch);
        manager.destroy();
    }
}

void CoarseVamanaIndex::initialize_query_scratch(uint32_t num_threads, uint32_t search_l, uint32_t indexing_l,
                                                    uint32_t r, uint32_t maxc, size_t dim)
{
    for (uint32_t i = 0; i < num_threads; i++)
    {
        auto scratch = new InMemQueryScratch(_data_size, search_l, indexing_l, r, maxc, dim, _data_store->get_aligned_dim(),
                                                _data_store->get_alignment_factor());
        _query_scratch.push(scratch);
    }
}

uint32_t CoarseVamanaIndex::calculate_entry_point()
{
    if (_use_half) {
        return _data_store->calculate_medoid_half();
    }
    return _data_store->calculate_medoid();
}

std::vector<uint32_t> CoarseVamanaIndex::get_init_ids()
{
    std::vector<uint32_t> init_ids;
    init_ids.reserve(1 + _num_frozen_pts);

    init_ids.emplace_back(_start);

    for (uint32_t frozen = (uint32_t)_max_points; frozen < _max_points + _num_frozen_pts; frozen++)
    {
        if (frozen != _start)
        {
            init_ids.emplace_back(frozen);
        }
    }

    return init_ids;
}

std::pair<uint32_t, uint32_t> CoarseVamanaIndex::iterate_to_fixed_point(const void *query, const uint32_t Lsize, const std::vector<uint32_t> &init_ids, 
                        InMemQueryScratch *scratch)
{
    std::vector<Neighbor> &expanded_nodes = scratch->pool();
    NeighborPriorityQueue &best_L_nodes = scratch->best_l_nodes();
    best_L_nodes.reserve(Lsize);
    tsl::robin_set<uint32_t> &inserted_into_pool_rs = scratch->inserted_into_pool_rs();
    boost::dynamic_bitset<> &inserted_into_pool_bs = scratch->inserted_into_pool_bs();
    std::vector<uint32_t> &id_scratch = scratch->id_scratch();
    std::vector<float> &dist_scratch = scratch->dist_scratch();
    if (id_scratch.size() != 0) {
        throw std::runtime_error("id_scratch is not empty!");
    }

    if (expanded_nodes.size() > 0 || id_scratch.size() > 0)
    {
        throw std::runtime_error("ERROR: Clear scratch space before passing.");
    }

    // Decide whether to use bitset or robin set to mark visited nodes
    auto total_num_points = _max_points + _num_frozen_pts;
    bool fast_iterate = total_num_points <= MAX_POINTS_FOR_USING_BITSET;
    if (fast_iterate)
    {
        if (inserted_into_pool_bs.size() < total_num_points)
        {
            // hopefully using 2X will reduce the number of allocations.
            auto resize_size =
                2 * total_num_points > MAX_POINTS_FOR_USING_BITSET ? MAX_POINTS_FOR_USING_BITSET : 2 * total_num_points;
            inserted_into_pool_bs.resize(resize_size);
        }
    }
    // Lambda to determine if a node has been visited
    auto is_not_visited = [this, fast_iterate, &inserted_into_pool_bs, &inserted_into_pool_rs](const uint32_t id) {
        return fast_iterate ? inserted_into_pool_bs[id] == 0
                            : inserted_into_pool_rs.find(id) == inserted_into_pool_rs.end();
    };

    // Initialize the candidate pool with starting points
    for (auto id : init_ids)
    {
        if (id >= _max_points + _num_frozen_pts)
        {
            throw std::runtime_error("Out of range loc found as an edge");
        }

        if (is_not_visited(id))
        {
            if (fast_iterate)
            {
                inserted_into_pool_bs[id] = 1;
            }
            else
            {
                inserted_into_pool_rs.insert(id);
            }
            float distance = _data_store->get_distance(query, id);
            Neighbor nn = Neighbor(id, distance);
            best_L_nodes.insert(nn);
        }
    }

    uint32_t hops = 0;
    uint32_t cmps = 0;

    while (best_L_nodes.has_unexpanded_node())
    {
        auto nbr = best_L_nodes.closest_unexpanded();
        auto n = nbr.id;
        expanded_nodes.emplace_back(nbr);
    

        // Find which of the nodes in des have not been visited before
        id_scratch.clear();

        dist_scratch.clear();

        _locks[n].lock();

        auto nbrs = _graph_store->get_neighbours(n);

        _locks[n].unlock();
        
        for (auto id : nbrs)
        {
            if (id >= _max_points + _num_frozen_pts)
            {
                throw std::runtime_error("id out of bound!!!");
            }

            if (is_not_visited(id))
            {
                id_scratch.push_back(id);
            }
        }
        
        // Mark nodes visited
        for (auto id : id_scratch)
        {
            if (fast_iterate)
            {
                inserted_into_pool_bs[id] = 1;
            }
            else
            {
                inserted_into_pool_rs.insert(id);
            }
        }

        if (dist_scratch.size() != 0)
        {
            throw std::runtime_error("dist_scratch not cleared!!!");
        }
        
        for (size_t m = 0; m < id_scratch.size(); ++m)
        {
            uint32_t id = id_scratch[m];

            if (m + 1 < id_scratch.size())
            {
                auto nextn = id_scratch[m + 1];
                _data_store->prefetch_vector(nextn);
            }

            dist_scratch.push_back(_data_store->get_distance(query, id));
        }
        cmps += (uint32_t)id_scratch.size();

        // Insert <id, dist> pairs into the pool of candidates
        for (size_t m = 0; m < id_scratch.size(); ++m)
        {
            best_L_nodes.insert(Neighbor(id_scratch[m], dist_scratch[m]));
        }
    }
    return std::make_pair(hops, cmps);
}

void CoarseVamanaIndex::occlude_list(const uint32_t location, std::vector<Neighbor> &pool, const float alpha,
                    const uint32_t degree, const uint32_t maxc, std::vector<uint32_t> &result,
                    InMemQueryScratch *scratch)
{
    if (pool.size() == 0)
        return;

    // Truncate pool at maxc and initialize scratch spaces
    if (!std::is_sorted(pool.begin(), pool.end())) {
        throw std::runtime_error("pool has not been sorted!!!");
    }
    if (result.size() != 0) {
        throw std::runtime_error("result should be empty!!!");
    }
    if (pool.size() > maxc)
        pool.resize(maxc);
    std::vector<float> &occlude_factor = scratch->occlude_factor();
    // occlude_list can be called with the same scratch more than once by
    // search_for_point_and_add_link through inter_insert.
    occlude_factor.clear();
    // Initialize occlude_factor to pool.size() many 0.0f values for correctness
    occlude_factor.insert(occlude_factor.end(), pool.size(), 0.0f);

    float cur_alpha = 1;
    while (cur_alpha <= alpha && result.size() < degree)
    {
        // used for MIPS, where we store a value of eps in cur_alpha to
        // denote pruned out entries which we can skip in later rounds.
        float eps = cur_alpha + 0.01f;

        for (auto iter = pool.begin(); result.size() < degree && iter != pool.end(); ++iter)
        {
            if (occlude_factor[iter - pool.begin()] > cur_alpha)
            {
                continue;
            }
            // Set the entry to float::max so that is not considered again
            occlude_factor[iter - pool.begin()] = std::numeric_limits<float>::max();

            if (iter->id != location)
            {
                result.push_back(iter->id);
            }

            // Update occlude factor for points from iter+1 to pool.end()
            for (auto iter2 = iter + 1; iter2 != pool.end(); iter2++)
            {
                auto t = iter2 - pool.begin();
                if (occlude_factor[t] > alpha)
                    continue;

                bool prune_allowed = true;
                
                if (!prune_allowed)
                    continue;

                float djk = _data_store->get_distance(iter2->id, iter->id);
                if (_data_store->get_measure_method() == IndexDistance::kMethodFloatSquaredEuclidean 
                    || _data_store->get_measure_method() == IndexDistance::kMethodHalfFloatSquaredEuclidean
                    || _data_store->get_measure_method() == IndexDistance::kMethodFloatEuclidean 
                    || _data_store->get_measure_method() == IndexDistance::kMethodFloatNormalizedEuclidean 
                    || _data_store->get_measure_method() == IndexDistance::kMethodFloatNormalizedSquaredEuclidean
                    || _data_store->get_measure_method() == IndexDistance::kMethodFloatCosine)
                {
                    occlude_factor[t] = (djk == 0) ? std::numeric_limits<float>::max()
                                                : std::max(occlude_factor[t], iter2->distance / djk);
                }
                else if (_data_store->get_measure_method() == IndexDistance::kMethodFloatInnerProduct
                        || _data_store->get_measure_method() == IndexDistance::kMethodHalfFloatInnerProduct)
                {
                    // Improvization for flipping max and min dist for MIPS
                    float x = -iter2->distance;
                    float y = -djk;
                    if (y > cur_alpha * x)
                    {
                        occlude_factor[t] = std::max(occlude_factor[t], eps);
                    }
                }
                else
                {
                    throw std::runtime_error("no such distance method");
                }
            }
        }
        cur_alpha *= 1.2f;
    }
}

void CoarseVamanaIndex::prune_neighbors(const uint32_t location, std::vector<Neighbor> &pool, const uint32_t range,
                        const uint32_t max_candidate_size, const float alpha,
                        std::vector<uint32_t> &pruned_list, InMemQueryScratch *scratch)
{
    if (pool.size() == 0)
    {
        // if the pool is empty, behave like a noop
        pruned_list.clear();
        return;
    }

    // sort the pool based on distance to query and prune it with occlude_list
    std::sort(pool.begin(), pool.end());
    pruned_list.clear();
    pruned_list.reserve(range);

    occlude_list(location, pool, alpha, range, max_candidate_size, pruned_list, scratch);
    if (pruned_list.size() > range) {
        throw std::runtime_error("pruned_list size over range!!!");
    }

    if (_saturate_graph && alpha > 1)
    {
        for (const auto &node : pool)
        {
            if (pruned_list.size() >= range)
                break;
            if ((std::find(pruned_list.begin(), pruned_list.end(), node.id) == pruned_list.end()) &&
                node.id != location)
                pruned_list.push_back(node.id);
        }
    }
}

void CoarseVamanaIndex::prune_neighbors(const uint32_t location, std::vector<Neighbor> &pool,
                        std::vector<uint32_t> &pruned_list, InMemQueryScratch *scratch)
{
    prune_neighbors(location, pool, _indexingRange, _indexingMaxC, _indexingAlpha, pruned_list, scratch);
}

void CoarseVamanaIndex::search_for_point_and_prune(int location, 
                                uint32_t Lindex,
                                std::vector<uint32_t> &pruned_list,
                                InMemQueryScratch *scratch)
{
    const std::vector<uint32_t> init_ids = get_init_ids();
    _data_store->get_vector(location, scratch->aligned_query());
    iterate_to_fixed_point(scratch->aligned_query(), Lindex, init_ids, scratch);

    auto &pool = scratch->pool();

    for (uint32_t i = 0; i < pool.size(); i++)
    {
        if (pool[i].id == (uint32_t)location)
        {
            pool.erase(pool.begin() + i);
            i--;
        }
    }

    if (pruned_list.size() > 0)
    {
        throw std::runtime_error("ERROR: non-empty pruned_list passed");
    }

    prune_neighbors(location, pool, pruned_list, scratch);

    if (pruned_list.empty()) {
        LOG_WARN("please check if entry point the same with init point");
    }

    if (_graph_store->get_total_points() != (_max_points + _num_frozen_pts)) {
        throw std::runtime_error("mismatch graph capacity!!!");
    }
}

void CoarseVamanaIndex::inter_insert(uint32_t n, std::vector<uint32_t> &pruned_list, const uint32_t range, InMemQueryScratch *scratch)
{
    const auto &src_pool = pruned_list;;

    for (auto des : src_pool)
    {
        // des_pool contains the neighbors of the neighbors of n
        std::vector<uint32_t> copy_of_neighbors;
        bool prune_needed = false;
        {
            std::lock_guard<std::mutex> guard(_locks[des]);
            auto &des_pool = _graph_store->get_neighbours(des);
            if (std::find(des_pool.begin(), des_pool.end(), n) == des_pool.end())
            {
                if (des_pool.size() < (uint64_t)(defaults::GRAPH_SLACK_FACTOR * range))
                {
                    // des_pool.emplace_back(n);
                    _graph_store->add_neighbour(des, n);
                    prune_needed = false;
                }
                else
                {
                    copy_of_neighbors.reserve(des_pool.size() + 1);
                    copy_of_neighbors = des_pool;
                    copy_of_neighbors.push_back(n);
                    prune_needed = true;
                }
            }
        } // des lock is released by this point

        if (prune_needed)
        {
            tsl::robin_set<uint32_t> dummy_visited(0);
            std::vector<Neighbor> dummy_pool(0);

            size_t reserveSize = (size_t)(std::ceil(1.05 * defaults::GRAPH_SLACK_FACTOR * range));
            dummy_visited.reserve(reserveSize);
            dummy_pool.reserve(reserveSize);

            for (auto cur_nbr : copy_of_neighbors)
            {
                if (dummy_visited.find(cur_nbr) == dummy_visited.end() && cur_nbr != des)
                {
                    float dist = _data_store->get_distance(des, cur_nbr);
                    dummy_pool.emplace_back(Neighbor(cur_nbr, dist));
                    dummy_visited.insert(cur_nbr);
                }
            }
            std::vector<uint32_t> new_out_neighbors;
            prune_neighbors(des, dummy_pool, new_out_neighbors, scratch);
            {
                std::lock_guard<std::mutex> guard(_locks[des]);

                _graph_store->set_neighbours(des, new_out_neighbors);
            }
        }
    }
}

void CoarseVamanaIndex::inter_insert(uint32_t n, std::vector<uint32_t> &pruned_list, InMemQueryScratch *scratch)
{
    inter_insert(n, pruned_list, _indexingRange, scratch);
}

void CoarseVamanaIndex::link()
{
    uint32_t num_threads = _indexingThreads;
    std::cout << "num_threads = " << num_threads << std::endl;
    /* visit_order is a vector that is initialized to the entire graph */
    std::vector<uint32_t> visit_order;
    std::vector<Neighbor> pool, tmp;
    tsl::robin_set<uint32_t> visited;
    visit_order.reserve(_nd + _num_frozen_pts);
    for (uint32_t i = 0; i < (uint32_t)_nd; i++)
    {
        visit_order.emplace_back(i);
    }

    // If there are any frozen points, add them all.
    for (uint32_t frozen = (uint32_t)_max_points; frozen < _max_points + _num_frozen_pts; frozen++)
    {
        visit_order.emplace_back(frozen);
    }

    // if there are frozen points, the first such one is set to be the _start
    if (_num_frozen_pts > 0)
        _start = (uint32_t)_max_points;
    else
        _start = calculate_entry_point();
    
    Timer link_timer;

    omp_set_num_threads(num_threads);

#pragma omp parallel for schedule(guided)
    for (int64_t node_ctr = 0; node_ctr < (int64_t)(visit_order.size()); node_ctr++)
    {
        auto node = visit_order[node_ctr];
        // Find and add appropriate graph edges
        ScratchStoreManager<InMemQueryScratch> manager(_query_scratch);
        auto scratch = manager.scratch_space();
        std::vector<uint32_t> pruned_list;
        search_for_point_and_prune(node, _indexingQueueSize, pruned_list, scratch);
        if (pruned_list.size() <= 0) {
            LOG_WARN("should find at least one neighbour (i.e frozen point acting as medoid)");
        }
        {
            std::lock_guard<std::mutex> guard(_locks[node]);

            _graph_store->set_neighbours(node, pruned_list);
        }

        inter_insert(node, pruned_list, scratch);

        if (node_ctr % 100000 == 0)
        {
            std::cout << "\r" << (100.0 * node_ctr) / (visit_order.size()) << "% of index build completed."<< std::endl;
        }
    }

    if (_nd > 0)
    {
        LOG_INFO("Starting final cleanup..");
    }

#pragma omp parallel for schedule(guided)
    for (int64_t node_ctr = 0; node_ctr < (int64_t)(visit_order.size()); node_ctr++)
    {
        auto node = visit_order[node_ctr];
        if (_graph_store->get_neighbours((uint32_t)node).size() > _indexingRange)
        {
            ScratchStoreManager<InMemQueryScratch> manager(_query_scratch);
            auto scratch = manager.scratch_space();

            tsl::robin_set<uint32_t> dummy_visited(0);
            std::vector<Neighbor> dummy_pool(0);
            std::vector<uint32_t> new_out_neighbors;

            for (auto cur_nbr : _graph_store->get_neighbours((uint32_t)node))
            {
                if (dummy_visited.find(cur_nbr) == dummy_visited.end() && cur_nbr != node)
                {
                    float dist = _data_store->get_distance(node, cur_nbr);
                    dummy_pool.emplace_back(Neighbor(cur_nbr, dist));
                    dummy_visited.insert(cur_nbr);
                }
            }
            prune_neighbors(node, dummy_pool, new_out_neighbors, scratch);

            _graph_store->clear_neighbours((uint32_t)node);
            _graph_store->set_neighbours((uint32_t)node, new_out_neighbors);
        }
    }
    if (_nd > 0)
    {
        std::cout << "done. Link time: " << ((double)link_timer.elapsed() / (double)1000000) << "s" << std::endl;
    }
}

void CoarseVamanaIndex::build_with_data_populated() {
    LOG_INFO("Starting index build with %lu points...", _nd);

    // resize the data_store to _nd
    this->_data_store->resize((uint32_t)_nd);

    uint32_t index_R = _indexingRange;
    uint32_t num_threads_index = _indexingThreads;
    uint32_t index_L = _indexingQueueSize;
    uint32_t maxc = _indexingMaxC;

    LOG_INFO("CoarseVamanaIndex parameters: R[%d], L[%d], maxc[%d], saturate[%s], Alpha[%f]", index_R, index_L, maxc, _saturate_graph ? "true" : "false", _indexingAlpha);

    if (_query_scratch.size() == 0)
    {
        initialize_query_scratch(5 + num_threads_index, index_L, index_L, index_R, maxc,
                                _data_store->get_aligned_dim());
    }

    link();

    size_t max = 0, min = SIZE_MAX, total = 0, cnt = 0;
    for (size_t i = 0; i < _nd; i++)
    {
        auto &pool = _graph_store->get_neighbours((uint32_t)i);
        max = std::max(max, pool.size());
        min = std::min(min, pool.size());
        total += pool.size();
        if (pool.size() < 2)
            cnt++;
    }

    float avg = (float)total / (float)(_nd + _num_frozen_pts);

    LOG_INFO("Index built with degree: max: %lu avg: %f min: %lu count(deg<2): %lu", max, avg, min, cnt);
    
    if (_saturate_graph && (max != (size_t)_indexingRange || avg != (float)_indexingRange || min != (size_t)_indexingRange)) {
        LOG_ERROR("Please check if data is highly duplicated");
        throw std::runtime_error("Duplicate Data Exception");
    }

    _has_built = true;
}

void CoarseVamanaIndex::build(const char *filename, const size_t num_points_to_load)
{
    if (!file_exists(filename))
    {
        LOG_ERROR("ERROR: data file %s does not exist.", filename);
        std::stringstream stream;
        stream << "ERROR: data file " << filename << " does not exist." << std::endl;
        throw new std::runtime_error(stream.str());
    }
    size_t file_num_points, file_dim;
    get_bin_metadata(filename, file_num_points, file_dim);
    if (file_num_points > _max_points)
    {
        std::stringstream stream;
        stream << "ERROR: Driver requests loading " << num_points_to_load << " points and file has " << file_num_points
            << " points, but "
            << "index can support only " << _max_points << " points as specified in constructor." << std::endl;
        throw new std::runtime_error(stream.str());
    }
    if (num_points_to_load > file_num_points)
    {
        std::stringstream stream;
        stream << "ERROR: Driver requests loading " << num_points_to_load << " points and file has only "
            << file_num_points << " points." << std::endl;
        throw new std::runtime_error(stream.str());
    }
    if (file_dim != _dim)
    {
        std::stringstream stream;
        stream << "ERROR: Driver requests loading " << _dim << " dimension,"
            << " but file has " << file_dim << " dimension." << std::endl;
        throw new std::runtime_error(stream.str());
    }
    {
        _nd = num_points_to_load;
        _data_store->populate_data(filename, 0U);
    }
    LOG_INFO("Using only first %lu from file.. ", num_points_to_load);
    build_with_data_populated();
}

// save the graph index on a file as an adjacency list. For each point,
// first store the number of neighbors, and then the neighbor list (each as
// 4 byte uint32_t)
size_t CoarseVamanaIndex::save_graph(std::string graph_file)
{
    return _graph_store->store(graph_file, _nd + _num_frozen_pts, _num_frozen_pts, _start);
}

size_t CoarseVamanaIndex::save_data(std::string data_file)
{
    // Note: at this point, either _nd == _max_points or any frozen points have
    // been temporarily moved to _nd, so _nd + _num_frozen_pts is the valid
    // location limit.
    return _data_store->save(data_file, (uint32_t)(_nd + _num_frozen_pts));
}

void CoarseVamanaIndex::save(const char *filename)
{
    std::string graph_file = std::string(filename);
    std::string data_file = std::string(filename) + ".data";

    Timer save_timer;

    delete_file(graph_file);
    save_graph(graph_file);
    delete_file(data_file);
    save_data(data_file);

    std::cout << "Time taken for save: " << save_timer.elapsed() / 1000000.0 << "s." << std::endl;
}

void CoarseVamanaIndex::dump(IndexPackage &index_package)
{
    _data_store->dump(index_package, (uint32_t)(_nd + _num_frozen_pts));
    _graph_store->dump(index_package, _nd + _num_frozen_pts, _num_frozen_pts, _start);
}

void CoarseVamanaIndex::reposition_points(uint32_t old_location_start, uint32_t new_location_start, uint32_t num_locations)
{
    if (num_locations == 0 || old_location_start == new_location_start)
    {
        return;
    }

    // Update pointers to the moved nodes. Note: the computation is correct even
    // when new_location_start < old_location_start given the C++ uint32_t
    // integer arithmetic rules.
    const uint32_t location_delta = new_location_start - old_location_start;

    std::vector<uint32_t> updated_neighbours_location;
    for (uint32_t i = 0; i < _max_points + _num_frozen_pts; i++)
    {
        auto &i_neighbours = _graph_store->get_neighbours((uint32_t)i);
        std::vector<uint32_t> i_neighbours_copy(i_neighbours.begin(), i_neighbours.end());
        for (auto &loc : i_neighbours_copy)
        {
            if (loc >= old_location_start && loc < old_location_start + num_locations)
                loc += location_delta;
        }
        _graph_store->set_neighbours(i, i_neighbours_copy);
    }

    // The [start, end) interval which will contain obsolete points to be
    // cleared.
    uint32_t mem_clear_loc_start = old_location_start;
    uint32_t mem_clear_loc_end_limit = old_location_start + num_locations;

    // Move the adjacency lists. Make sure that overlapping ranges are handled
    // correctly.
    if (new_location_start < old_location_start)
    {
        // New location before the old location: copy the entries in order
        // to avoid modifying locations that are yet to be copied.
        for (uint32_t loc_offset = 0; loc_offset < num_locations; loc_offset++)
        {
            assert(_graph_store->get_neighbours(new_location_start + loc_offset).empty());
            _graph_store->swap_neighbours(new_location_start + loc_offset, old_location_start + loc_offset);
        }
        // If ranges are overlapping, make sure not to clear the newly copied
        // data.
        if (mem_clear_loc_start < new_location_start + num_locations)
        {
            // Clear only after the end of the new range.
            mem_clear_loc_start = new_location_start + num_locations;
        }
    }
    else
    {
        // Old location after the new location: copy from the end of the range
        // to avoid modifying locations that are yet to be copied.
        for (uint32_t loc_offset = num_locations; loc_offset > 0; loc_offset--)
        {
            assert(_graph_store->get_neighbours(new_location_start + loc_offset - 1u).empty());
            _graph_store->swap_neighbours(new_location_start + loc_offset - 1u, old_location_start + loc_offset - 1u);
        }

        // If ranges are overlapping, make sure not to clear the newly copied
        // data.
        if (mem_clear_loc_end_limit > new_location_start)
        {
            // Clear only up to the beginning of the new range.
            mem_clear_loc_end_limit = new_location_start;
        }
    }
    _data_store->move_vectors(old_location_start, new_location_start, num_locations);
}

void CoarseVamanaIndex::resize(size_t new_max_points)
{
    const size_t new_internal_points = new_max_points + _num_frozen_pts;
    auto start = std::chrono::high_resolution_clock::now();

    _data_store->resize((uint32_t)new_internal_points);
    _graph_store->resize_graph(new_internal_points);
    _locks = std::vector<std::mutex>(new_internal_points);

    if (_num_frozen_pts != 0)
    {
        reposition_points((uint32_t)_max_points, (uint32_t)new_max_points, (uint32_t)_num_frozen_pts);
        _start = (uint32_t)new_max_points;
    }

    _max_points = new_max_points;

    auto stop = std::chrono::high_resolution_clock::now();
    LOG_INFO("Resizing took: %lfs", std::chrono::duration<double>(stop - start).count());
}

void CoarseVamanaIndex::get_data_vec(const uint32_t i, void *dest) const
{
    _data_store->get_vector(i, dest);
}

void CoarseVamanaIndex::set_data_vec(const uint32_t loc, const void *const vector)
{
    if (_nd > _max_points) 
    {
        resize(_max_points * defaults::MEM_ALLOC_EXPAND_FACTOR);
    }

    _data_store->set_vector(loc, vector);
    
    if (_id_set.find(loc) == _id_set.end())
    {
        _id_set.insert(_nd);
        _nd++;
    }
}

size_t CoarseVamanaIndex::load_data(std::string filename)
{
    size_t file_dim, file_num_points;
    if (!file_exists(filename))
    {
        std::stringstream stream;
        stream << "ERROR: data file " << filename << " does not exist." << std::endl;
        throw std::runtime_error(stream.str());
    }
    get_bin_metadata(filename, file_num_points, file_dim);
    if (file_dim != _dim)
    {
        std::stringstream stream;
        stream << "here ERROR: Driver requests loading " << _dim << " dimension,"
            << " but file has " << file_dim << " dimension." << std::endl;
        throw std::runtime_error(stream.str());
    }

    if (file_num_points > _max_points + _num_frozen_pts)
    {
        // update and tag lock acquired in load() before calling load_data
        resize(file_num_points - _num_frozen_pts);
    }

    _data_store->load(filename); // offset == 0.

    return file_num_points;
}

size_t CoarseVamanaIndex::load_graph(std::string filename, size_t expected_num_points)
{
    auto res = _graph_store->load(filename, expected_num_points);
    _start = std::get<1>(res);
    _num_frozen_pts = std::get<2>(res);
    return std::get<0>(res);
}

void CoarseVamanaIndex::load(const char *filename, uint32_t num_threads, uint32_t search_l)
{
    _has_built = true;

    size_t graph_num_pts = 0, data_file_num_pts = 0;

    std::string mem_index_file(filename);
    std::string labels_file = mem_index_file + "_labels.txt";
    std::string labels_to_medoids = mem_index_file + "_labels_to_medoids.txt";
    std::string labels_map_file = mem_index_file + "_labels_map.txt";
    std::string data_file = std::string(filename) + ".data";
    std::string tags_file = std::string(filename) + ".tags";
    std::string delete_set_file = std::string(filename) + ".del";
    std::string graph_file = std::string(filename);
    data_file_num_pts = load_data(data_file);
    graph_num_pts = load_graph(graph_file, data_file_num_pts);

    if (data_file_num_pts != graph_num_pts)
    {
        std::stringstream stream;
        stream << "ERROR: When loading index, loaded " << data_file_num_pts << " points from datafile, "
            << graph_num_pts << " from graph in constructor." << std::endl;
        throw std::runtime_error(stream.str());
    }

    if (_query_scratch.size() == 0)
    {
        initialize_query_scratch(num_threads, search_l, search_l, (uint32_t)_graph_store->get_max_range_of_graph(),
                                _indexingMaxC, _dim);
    }
}

std::pair<uint32_t, uint32_t> CoarseVamanaIndex::search(const void *query, const size_t K, const uint32_t L,
                                                        uint64_t *indices, float *distance)
{
    ScratchStoreManager<InMemQueryScratch> manager(_query_scratch);
    auto scratch = manager.scratch_space();

    const std::vector<uint32_t> init_ids = get_init_ids();

    _data_store->preprocess_query(query, _data_store->get_dims(), scratch->aligned_query());

    auto retval = iterate_to_fixed_point(scratch->aligned_query(), L, init_ids, scratch);

    NeighborPriorityQueue &best_L_nodes = scratch->best_l_nodes();

    size_t pos = 0;
    for (size_t i = 0; i < best_L_nodes.size(); ++i)
    {
        if (best_L_nodes[i].id < _max_points)
        {
            indices[pos] = (uint64_t)best_L_nodes[i].id;
            distance[pos] = (float)best_L_nodes[i].distance;
            pos++;
        }
        if (pos == K)
        break;
    }

    if (pos < K)
    {
        LOG_ERROR("Found pos: %lu fewer than K elements %lu for query", pos, K);
    }

    return retval;
}

size_t CoarseVamanaIndex::get_active_num()
{
    return _nd;
}

MERCURY_NAMESPACE_END(core);