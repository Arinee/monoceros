# pragma once

#include "src/core/framework/index_framework.h"
#include "src/core/algorithm/index.h"
#include "coarse_vamana_index.h"
#include "src/core/algorithm/centroid_resource.h"
#include "src/core/algorithm/centroid_resource_manager.h"
#include "src/core/algorithm/group_manager.h"
#include "src/core/algorithm/ivf_pq/query_distance_matrix.h"
#include "src/core/algorithm/hdfs_file_wrapper.h"
#include "src/core/utils/vamana/aligned_file_reader.h"
#include "src/core/utils/vamana/percentile_stats.h"
#include "src/core/algorithm/ivf_pq/pq_dist_scorer.h"
#include "src/core/algorithm/query_info.h"

MERCURY_NAMESPACE_BEGIN(core);

typedef std::unique_ptr<char[]> MatrixPointer;

class DiskVamanaIndex : public Index
{
public:
    DiskVamanaIndex();

    ~DiskVamanaIndex();

    typedef std::shared_ptr<DiskVamanaIndex> Pointer;

    virtual int Create(IndexParams& index_params);

    int Load(const void* data, size_t size) override;

    int LoadDiskIndex(const std::string& disk_index_path, const std::string& medoids_data_path);

    void cache_bfs_levels(uint64_t num_nodes_to_cache, std::vector<uint32_t> &node_list);

    void load_cache_list(std::vector<uint32_t> &node_list);

    void cached_beam_search(const void *query, const uint64_t k_search, const uint64_t l_search,
                                                 uint64_t *indices, float *distances, const uint64_t beam_width,
                                                 QueryStats *stats, const uint32_t io_limit = std::numeric_limits<uint32_t>::max());

    void cached_beam_search_half(const void *query, const void *query_raw, const uint64_t k_search, const uint64_t l_search,
                                                 uint64_t *indices, float *distances, const uint64_t beam_width,
                                                 QueryStats *stats, const uint32_t io_limit = std::numeric_limits<uint32_t>::max());
    
    size_t MemQuota2DocCount(size_t mem_quota, size_t elem_size) const override {
        //TODO
        return 0;
    }

    bool IsFull() const override {
        return false; 
    }

    int64_t UsedMemoryInCurrent() const override {
        //TODO
        return 0;
    }

    uint64_t GetMaxDocNum() const override {
        return max_doc_num_;
    }

    size_t GetDocNum() const override {
        return doc_num_;
    }

    // no use
    float GetRankScore(docid_t doc_id) override { return 0;};

    int BaseIndexAdd(docid_t doc_id, pk_t pk, const void *val, size_t len);

    int AddOriData(const void *val);

    int AddRawData(const void *val);

    int AddShardData(int shard, docid_t doc_id, const void *val);

    int PartitionBaseIndexAdd(docid_t doc_id, pk_t pk, const QueryInfo& query_info, const QueryInfo& query_info_raw);

    int PartitionBaseIndexRandAdd(docid_t doc_id, pk_t pk, const QueryInfo& query_info, const QueryInfo& query_info_raw);

    int PartitionBaseIndexDump();

    int ShardDataDump(std::string &shardToken, std::string &path);

    int ShardIndexBuildAndDump(std::string &shardToken, std::string &indexPath);

    void BuildPqIndexFromFile(std::string filename);

    int PQIndexAdd(docid_t doc_id, const void *val);

    void GetBaseVec(docid_t doc_id, void* dest);

    size_t GetBaseVecNum();

    void BuildMemIndex();

    void BuildAndDumpPartitionIndex();

    void MergeShardIndex(std::string &path_medoids, size_t* size_medoids, 
                                      const std::vector<std::string> &shardIndexFiles, 
                                      const std::vector<std::string> &shardIdmapFiles);

    void MergePartitionIndex(std::string &path_medoids, size_t* size_medoids);

    void DumpMemLocal(const std::string filename);

    int Dump(const void*& data, size_t& size) override;

    int DumpPqLocal(const std::string& pq_file_name);

    void CreateDiskLayout(const std::string base_file, const std::string mem_index_file, const std::string output_file);

    float compare(const void *a, const void *b, uint32_t length);

protected:
    bool InitPqCentroidMatrix(const IndexParams& param);

    bool InitVamanaCentroidMatrix(const std::string& centroid_dir);

    MatrixPointer DoLoadCentroidMatrix(const std::string& file_path, size_t dimension,
                                       size_t element_size, size_t& centroid_size) const;

    bool StrToValue(const std::string& source, void* value) const;

private:
    // (default is 64): the degree of the graph index, typically between 60 and 150. 
    // Larger R will result in larger indices and longer indexing times, but better search quality. 
    uint32_t R_;

    // (default is 100): the size of search list during index build. 
    // Typical values are between 75 to 200. 
    // Larger values will take more time to build but result in indices that provide higher recall for the same search complexity. 
    // Use a value for L value that is at least the value of R unless you need to build indices really quickly and can somewhat compromise on quality. 
    uint32_t L_;

    float alpha_;

    bool is_saturated_;

    uint32_t max_occlusion_;

    // (default is 1): num of threads to build coarse vamana index
    uint32_t T_;

    // doc num
    uint64_t doc_num_;

    // max doc num
    uint64_t max_doc_num_;

    // max shard data num
    uint64_t max_shard_data_num_;

    uint64_t aligned_dim_ = 0;

    uint64_t data_dim_ = 0;

    uint16_t data_size_ = 0;

    // base Vamana index structure
    std::unique_ptr<CoarseVamanaIndex> coarseVamanaIndex_;

    // to calculate centroid L2 distance
    IndexMeta index_meta_L2_;

    int16_t GetOrDefault(const IndexParams& params, const std::string& key, const uint16_t default_value);

    GroupManager group_manager_;

    // for pq
    ArrayProfile pq_code_profile_;

    std::vector<char> pq_code_base_;

    std::string integrate_matrix_;

    // 中心点管理类：每个类目的中心点信息 + 0类目还存pq分段中心点信息
    CentroidResourceManager centroid_resource_manager_;

    bool _pq_table_populated;

    // graph has one entry point by default,
    // we can optionally have multiple starting points
    uint32_t *_medoids = nullptr;

    // defaults to 1
    size_t _num_medoids;

    // by default, it is empty. If there are multiple
    // centroids, we pick the medoid corresponding to the
    // closest centroid as the starting point of search
    float *_centroid_data = nullptr;

    half_float::half *_centroid_half_data = nullptr;

    // nhood_cache; the uint32_t in nhood_Cache are offsets into nhood_cache_buf
    unsigned *_nhood_cache_buf = nullptr;
    tsl::robin_map<uint32_t, std::pair<uint32_t, uint32_t *>> _nhood_cache;
    std::unique_ptr<tsl::robin_map<uint32_t, std::pair<uint32_t, uint32_t *>>> nhood_cache_;

    // coord_cache; The second parameter in coord_cache are offsets into coord_cache_buf
    std::vector<float> _coord_cache_buf;

    std::unique_ptr<tsl::robin_map<uint32_t, float *>> coord_cache_;

    bool _use_half;

    std::vector<half_float::half> _coord_cache_buf_half;

    std::unique_ptr<tsl::robin_map<uint32_t, half_float::half *>> coord_cache_half_;

    // index info for multi-node sectors
    // nhood of node `i` is in sector: [i / nnodes_per_sector]
    // offset in sector: [(i % nnodes_per_sector) * max_node_len]
    //
    // index info for multi-sector nodes
    // nhood of node `i` is in sector: [i * DIV_ROUND_UP(_max_node_len, SECTOR_LEN)]
    // offset in sector: [0]
    //
    // Common info
    // coords start at ofsset
    // #nbrs of node `i`: *(unsigned*) (offset + disk_bytes_per_point)
    // nbrs of node `i` : (unsigned*) (offset + disk_bytes_per_point + 1)
    uint64_t _max_node_len = 0;

    uint64_t _nnodes_per_sector = 0; // 0 for multi-sector nodes, >0 for multi-node sectors

    uint64_t _max_degree = 0;

    uint64_t _disk_bytes_per_point = 0; // Number of bytes

    // setting up concept of frozen points in disk index for streaming-DiskANN
    uint64_t _num_frozen_points = 0;

    bool _reorder_data_exists = false;

    std::shared_ptr<AlignedFileReader> reader;

    ConcurrentQueue<SSDThreadData *> _thread_data;

    // for partition build
    uint32_t _k_base = 0;

    uint32_t _center_num = 0;

    std::unique_ptr<size_t[]> _shard_counts;

    std::vector<cached_ofstream *> _cached_shard_data_writer;

    std::vector<cached_ofstream *> _cached_shard_idmap_writer;

    cached_ofstream *_cached_ori_data_writer;

    uint32_t _ori_data_num;

    cached_ofstream *_cached_raw_data_writer;

    IndexDistance::Measure _measure;

    IndexDistance::Methods _method;
    
public:
    std::string _partition_prefix = "";

private:
    MatrixPointer NullMatrix() const;

    void use_medoids_data_as_centroids();

    std::vector<bool> read_nodes(const std::vector<uint32_t> &node_ids,
                                    std::vector<float *> &coord_buffers, 
                                    std::vector<half_float::half *> &coord_buffers_half,
                                    std::vector<std::pair<uint32_t, uint32_t *>> &nbr_buffers);

    // ptr to start of the node
    char *offset_to_node(char *sector_buf, uint64_t node_id);

    // returns region of `node_buf` containing [NNBRS][NBR_ID(uint32_t)]
    uint32_t *offset_to_node_nhood(char *node_buf);

    // returns region of `node_buf` containing [COORD(T)]
    float *offset_to_node_coords(char *node_buf);

    half_float::half *offset_to_node_coords_half(char *node_buf);

    // sector # on disk where node_id is present with in the graph part
    uint64_t get_node_sector(uint64_t node_id);

    // compute pq dists
    void compute_pq_dists(const void *query, const uint32_t *ids, const uint64_t n_ids, float *dists_out);

public:
    const CentroidResource& GetPqCentroidResource() const{
        return centroid_resource_manager_.GetCentroidResource(0);
    }

    CentroidResource& GetPqCentroidResource() {
        return centroid_resource_manager_.GetCentroidResource(0);
    }

    CentroidResource& GetVamanaCentroidResource() {
        return centroid_resource_manager_.GetCentroidResource(1);
    }

    std::shared_ptr<AlignedFileReader> GetAlignedReader() {
        return reader;
    }

    int GetCenterNum() {
        return _center_num;
    }

    bool IsPqTablePopulated() {
        return _pq_table_populated;
    }

};

MERCURY_NAMESPACE_END(core);