#pragma once

#include <sstream>
#include <vector>
#include <boost/dynamic_bitset.hpp>
#include "tsl/robin_set.h"
#include "tsl/robin_map.h"
#include "tsl/sparse_map.h"
#include "src/core/common/common.h"
#include "utils.h"
#include "neighbor.h"
#include "src/core/algorithm/vamana/vamana_defaults.h"
#include "src/core/utils/vamana/concurrent_queue.h"
#include <memory.h>
#include "aligned_file_reader.h"

#define NUM_PQ_BITS 8
#define NUM_PQ_CENTROIDS (1 << NUM_PQ_BITS)
#define MAX_PQ_CHUNKS 512

MERCURY_NAMESPACE_BEGIN(core);

//
// Scratch space for PQ
//
struct PQScratch
{
    float *aligned_pqtable_dist_scratch = nullptr; // MUST BE AT LEAST [256 * NCHUNKS]
    float *aligned_dist_scratch = nullptr;         // MUST BE AT LEAST  MAX_DEGREE
    uint16_t *aligned_pq_coord_scratch = nullptr;   // MUST BE AT LEAST  [N_CHUNKS * MAX_DEGREE]

    PQScratch(size_t graph_degree, size_t aligned_dim)
    {
        alloc_aligned((void **)&aligned_pq_coord_scratch,
                               (size_t)graph_degree * (size_t)MAX_PQ_CHUNKS * sizeof(uint8_t), 256);
        alloc_aligned((void **)&aligned_pqtable_dist_scratch, 256 * (size_t)MAX_PQ_CHUNKS * sizeof(float),
                               256);
        alloc_aligned((void **)&aligned_dist_scratch, (size_t)graph_degree * sizeof(float), 256);
    }

    ~PQScratch()
    {
        aligned_free(aligned_pqtable_dist_scratch);
        aligned_free(aligned_dist_scratch);
        aligned_free(aligned_pq_coord_scratch);
    }
};

//
// Scratch space for in-memory index based search
//
class InMemQueryScratch
{
  public:
    InMemQueryScratch(uint16_t data_size, uint32_t search_l, uint32_t indexing_l, uint32_t r, 
                        uint32_t maxc, size_t dim, size_t aligned_dim, size_t alignment_factor)
        : _L(0), _R(r), _maxc(maxc)
    {
        if (search_l == 0 || indexing_l == 0 || r == 0 || dim == 0)
        {
            std::stringstream stream;
            stream << "In InMemQueryScratch, one of search_l = " << search_l << ", indexing_l = " << indexing_l
            << ", dim = " << dim << " or r = " << r << " is zero." << std::endl;
            throw new std::runtime_error(stream.str());
        }

        alloc_aligned(((void **)&_aligned_query), aligned_dim * data_size, alignment_factor * data_size);
        memset(_aligned_query, 0, aligned_dim * data_size);

        _occlude_factor.reserve(maxc);
        _inserted_into_pool_bs = new boost::dynamic_bitset<>();
        _id_scratch.reserve((size_t)std::ceil(1.5 * defaults::GRAPH_SLACK_FACTOR * _R));
        _dist_scratch.reserve((size_t)std::ceil(1.5 * defaults::GRAPH_SLACK_FACTOR * _R));

        resize_for_new_L(std::max(search_l, indexing_l));
    }

    void resize_for_new_L(uint32_t new_l)
    {
        if (new_l > _L)
        {
            _L = new_l;
            _pool.reserve(3 * _L + _R);
            _best_l_nodes.reserve(_L);

            _inserted_into_pool_rs.reserve(20 * _L);
        }
    }

    void clear()
    {
        _pool.clear();
        _best_l_nodes.clear();
        _occlude_factor.clear();

        _inserted_into_pool_rs.clear();
        _inserted_into_pool_bs->reset();

        _id_scratch.clear();
        _dist_scratch.clear();

        _expanded_nodes_set.clear();
        _expanded_nghrs_vec.clear();
        _occlude_list_output.clear();
    }

    ~InMemQueryScratch()
    {
        if (_aligned_query != nullptr)
        {
            aligned_free(_aligned_query);
        }

        delete _inserted_into_pool_bs;
    }

    inline uint32_t get_L()
    {
        return _L;
    }

    inline uint32_t get_R()
    {
        return _R;
    }

    inline uint32_t get_maxc()
    {
        return _maxc;
    }

    inline void *aligned_query()
    {
        return _aligned_query;
    }

    inline std::vector<Neighbor> &pool()
    {
        return _pool;
    }

    inline NeighborPriorityQueue &best_l_nodes()
    {
        return _best_l_nodes;
    }

    inline std::vector<float> &occlude_factor()
    {
        return _occlude_factor;
    }
    
    inline tsl::robin_set<uint32_t> &inserted_into_pool_rs()
    {
        return _inserted_into_pool_rs;
    }

    inline boost::dynamic_bitset<> &inserted_into_pool_bs()
    {
        return *_inserted_into_pool_bs;
    }

    inline std::vector<uint32_t> &id_scratch()
    {
        return _id_scratch;
    }

    inline std::vector<float> &dist_scratch()
    {
        return _dist_scratch;
    }

    inline tsl::robin_set<uint32_t> &expanded_nodes_set()
    {
        return _expanded_nodes_set;
    }

    inline std::vector<Neighbor> &expanded_nodes_vec()
    {
        return _expanded_nghrs_vec;
    }

    inline std::vector<uint32_t> &occlude_list_output()
    {
        return _occlude_list_output;
    }

  private:
    uint32_t _L;
    uint32_t _R;
    uint32_t _maxc;

    void *_aligned_query = nullptr;

    // _pool stores all neighbors explored from best_L_nodes.
    // Usually around L+R, but could be higher.
    // Initialized to 3L+R for some slack, expands as needed.
    std::vector<Neighbor> _pool;

    // _best_l_nodes is reserved for storing best L entries
    // Underlying storage is L+1 to support inserts
    NeighborPriorityQueue _best_l_nodes;

    // _occlude_factor.size() >= pool.size() in occlude_list function
    // _pool is clipped to maxc in occlude_list before affecting _occlude_factor
    // _occlude_factor is initialized to maxc size
    std::vector<float> _occlude_factor;

    // Capacity initialized to 20L
    tsl::robin_set<uint32_t> _inserted_into_pool_rs;

    // Use a pointer here to allow for forward declaration of dynamic_bitset
    // in public headers to avoid making boost a dependency for clients
    // of DiskANN.
    boost::dynamic_bitset<> *_inserted_into_pool_bs;

    // _id_scratch.size() must be > R*GRAPH_SLACK_FACTOR for iterate_to_fp
    std::vector<uint32_t> _id_scratch;

    // _dist_scratch must be > R*GRAPH_SLACK_FACTOR for iterate_to_fp
    // _dist_scratch should be at least the size of id_scratch
    std::vector<float> _dist_scratch;

    //  Buffers used in process delete, capacity increases as needed
    tsl::robin_set<uint32_t> _expanded_nodes_set;
    std::vector<Neighbor> _expanded_nghrs_vec;
    std::vector<uint32_t> _occlude_list_output;
};

class SSDQueryScratch
{
  public:
    SSDQueryScratch(uint16_t data_size, size_t aligned_dim, size_t visited_reserve) 
    {
        size_t coord_alloc_size = ROUND_UP(data_size * aligned_dim, 256);

        alloc_aligned((void **)&coord_scratch, coord_alloc_size, 256);
        alloc_aligned((void **)&sector_scratch, defaults::MAX_N_SECTOR_READS * defaults::SECTOR_LEN,
                            defaults::SECTOR_LEN);
        alloc_aligned((void **)&aligned_query_T, aligned_dim * data_size, 8 * data_size);

        _pq_scratch = new PQScratch(defaults::MAX_GRAPH_DEGREE, aligned_dim);

        memset(coord_scratch, 0, coord_alloc_size);
        memset(aligned_query_T, 0, aligned_dim * data_size);

        visited.reserve(visited_reserve);
        full_retset.reserve(visited_reserve);
    }
    
    ~SSDQueryScratch()
    {
        aligned_free((void *)coord_scratch);
        aligned_free((void *)sector_scratch);
        aligned_free((void *)aligned_query_T);

        delete _pq_scratch;
    }

    void reset()
    {
        sector_idx = 0;
        visited.clear();
        retset.clear();
        full_retset.clear();
    }

  public:
    void *coord_scratch = nullptr; // MUST BE AT LEAST [data_size * data_dim]

    char *sector_scratch = nullptr; // MUST BE AT LEAST [MAX_N_SECTOR_READS * SECTOR_LEN]

    size_t sector_idx = 0;          // index of next [SECTOR_LEN] scratch to use

    void *aligned_query_T = nullptr;

    PQScratch *_pq_scratch;

    tsl::robin_set<size_t> visited;
    NeighborPriorityQueue retset;
    std::vector<Neighbor> full_retset;
};

class SSDThreadData
{
  public:
    SSDThreadData(uint16_t data_size, size_t aligned_dim, uint64_t visited_reserve = 4096)
    : scratch(data_size, aligned_dim, visited_reserve) {}

    void clear()
    {
        scratch.reset();
    }
    
  public:
    SSDQueryScratch scratch;
    IOContext ctx;
};

//
// Class to avoid the hassle of pushing and popping the query scratch.
//
template <typename T> 
class ScratchStoreManager
{
  public:
    ScratchStoreManager(ConcurrentQueue<T *> &query_scratch) : _scratch_pool(query_scratch)
    {
        _scratch = query_scratch.pop();
        while (_scratch == nullptr)
        {
            query_scratch.wait_for_push_notify();
            _scratch = query_scratch.pop();
        }
    }
    T *scratch_space()
    {
        return _scratch;
    }

    ~ScratchStoreManager()
    {
        _scratch->clear();
        _scratch_pool.push(_scratch);
        _scratch_pool.push_notify_all();
    }

    void destroy()
    {
        while (!_scratch_pool.empty())
        {
            auto scratch = _scratch_pool.pop();
            while (scratch == nullptr)
            {
                _scratch_pool.wait_for_push_notify();
                scratch = _scratch_pool.pop();
            }
            delete scratch;
        }
    }

  private:
    T *_scratch;
    ConcurrentQueue<T *> &_scratch_pool;
    ScratchStoreManager(const ScratchStoreManager<T> &);
    ScratchStoreManager &operator=(const ScratchStoreManager<T> &);
};

MERCURY_NAMESPACE_END(core);