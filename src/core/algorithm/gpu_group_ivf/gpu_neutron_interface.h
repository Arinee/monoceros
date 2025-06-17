/*********************************************************************
 * $Author: anduo $
 *
 * $LastChangedBy: anduo $
 *
 * $LastChangedDate: 2023-03-13 17:00 $
 *
 * $Id: gpu_linear_searcher.cuh 2023-01-16 17:00 anduo $
 *
 ********************************************************************/
#ifdef ENABLE_GPU_IN_MERCURY_

#ifndef INTERFACE_GPU_INDEX_SEARCHER_H_
#define INTERFACE_GPU_INDEX_SEARCHER_H_

#include <stdint.h>
#include <vector>

namespace neutron {
namespace gpu {

class GpuKernelIndex;
class GpuResourcesManager;

class GpuDataOffset
{
public:
    void PrintStats();

public:
    uint64_t query_vec = 0; // 0
    uint64_t docid_list = 0;
    uint64_t sort_docid_list = 0;
    uint64_t calcblock_start = 0;
    uint64_t calcblock_end = 0;
    uint64_t calcblock_query_index = 0;
    uint64_t phase1_sortblock_start = 0;
    uint64_t phase1_sortblock_end = 0;
    uint64_t phase1_sortblock_query_offset = 0;
    uint64_t phase2_sortblock_start = 0;
    uint64_t phase2_sortblock_end = 0;
    uint64_t cpu_result = 0;
    uint64_t memcpy_data_bytes = 0;
    uint64_t distance_bytes = 0;
    uint64_t sort_tmp_bytes = 0;
    uint64_t result_bytes = 0;
    uint64_t allocated_bytes = 0;
    uint64_t gpu_need_bytes = 0;
};

class GpuDataParam
{
public:
    GpuDataParam(uint32_t dim, uint32_t sort_mode, uint32_t element_size = 4);
    void PrintStats();

public:
    uint32_t dim = 0;
    uint32_t query_num = 0;
    uint32_t doc_num = 0;
    uint32_t aligned_doc_num = 0;
    uint32_t calc_block_num = 0;
    uint32_t phase1_sortblock_num = 0;
    uint32_t phase2_sortblock_num = 0;
    uint32_t max_topk = 0;
    uint32_t max_len = 0;
    uint32_t sort_mode = 0; // 0: 正常一起排序，1: 按传入的id list排序
    uint32_t element_size = 4; // 默认是float
};

class QueryDataParam
{
public:
    QueryDataParam(uint32_t doc_num, uint32_t topk, uint32_t phase2_sortblock_num);

public:
    uint32_t doc_num = 0;
    uint32_t aligned_doc_num = 0;
    uint32_t calc_block_num = 0;
    uint32_t phase1_sortblock_num = 0;
    uint32_t phase2_sortblock_num = 0; // sort_mode = 1时需要传入
    uint32_t topk = 0;
};

void AddQueryParam(GpuDataParam &gpu_data_param, QueryDataParam &query_data_param);
void CalcGpuData(GpuDataParam &gpu_data_param, GpuDataOffset &gpu_data_offset);
void FillGeneralData(char *cpu_data, GpuDataParam &gpu_data_param, GpuDataOffset &gpu_data_offset,
                     std::vector<QueryDataParam> &query_data_params,
                     std::vector<std::vector<uint32_t>> &sort_list_doc_nums);

class NeutronManagerInterface
{
public:
    NeutronManagerInterface(size_t initial_small_num = 30, size_t initial_large_num = 2,
                            size_t small_size = 60 * 1024 * 1024, size_t large_size = 200 * 1024 * 1024,
                            size_t pinned_mem_size = 4 * 1024 * 1024, size_t threshold = 60 * 1024 * 1024);
    ~NeutronManagerInterface();

    GpuResourcesManager *gpu_resources_manager_;
};

class NeutronIndexInterface
{
public:
    enum CALCULATORTYPE
    {
        FLOAT,
        HALF,
        PQ,
        DIRECTHALF,
        RPQ
    };
    NeutronIndexInterface(CALCULATORTYPE calculator_type, int dim, int element_size, int device_no = 0);

    ~NeutronIndexInterface();

    /**
     * @brief add feature to gpu global memory
     * @param feature: the frist addr of feature
     * @param num: the num of our feature(vector)
     * @return 0:success -1:failed
     */
    int Add(const char *feature, int num);

    /**
     * @brief add table to gpu global memory
     * @param feature: the frist addr of feature
     * @param num: the num of our feature(vector)
     * @return 0:success -1:failed
     */
    int AddTable(const char *feature, int num);

    /**
     * @param in_queries: query feature
     * @param out_distances: result distance
     * @param out_labels: result labels
     * @param in_indices: selected doc indices
     * @param qnum: query num
     * @topk: return topk result
     */
    int Search(const void *in_queries, std::vector<float> &out_distances, std::vector<int> &out_labels,
               std::vector<uint32_t> &in_indices, int qnum, int topk, const void *pq_codebook = nullptr,
               uint32_t pq_codes_count = 0);
    
    /**
     * @param out_distances: result distance
     * @param out_labels: result labels
     * @param in_indices: selected doc indices
     * @param cent_indices: level1 cent indices
     * @param in_distances: level1 cent dists
     * @param topk: return topk result
     * @param pq_codebook: PQ_QDM(IpValArray)
     * @param pq_codes_count: QDM size
     */
    int SearchTable(std::vector<float> &out_distances, std::vector<int> &out_labels,
                    std::vector<uint32_t> &in_indices, std::vector<uint32_t> &cent_indices,
                    std::vector<float> &in_distances, int topk,
                    const void *pq_codebook, uint32_t pq_codes_count);
    
    /**
     * @param in_queries: query feature
     * @param out_distances: result distance
     * @param out_labels: result labels
     * @param in_indices: selected doc indices
     * @param qnum: query num
     * @param topk: return topk result
     * @param group_num: group num
     * @param merged_group_doc_nums: doc num in each group
     */
    int SearchForCate(const void *in_queries, std::vector<float> &out_distances, std::vector<int> &out_labels,
                      std::vector<uint32_t> &in_indices, int qnum, int topk, int group_num,
                      std::vector<size_t> &merged_group_doc_nums);

    /**
     * @param task_num: task_num
     * @param sort_block_num: sort_block_num (used in sort)
     * @param max_topk: max_topk
     * @param cpu_batch_data: cpu batch data
     * @param data_offset: data offset
     * @param allocated_bytes: allocated bytes of cpu batch data
     */
    int SearchBatch(uint32_t task_num, uint32_t sort_block_num, uint32_t max_topk, uint32_t max_len,
                    char *cpu_batch_data, uint64_t *data_offset, uint64_t allocated_bytes);

    /**
     * @param task_num: task_num
     * @param batch_group_num: batch_group_num (used in sort)
     * @param max_topk: max_topk
     * @param cpu_batch_data: cpu batch data
     * @param data_offset: data offset
     * @param allocated_bytes: allocated bytes of cpu batch data
     * @param enable_rt: enable rt
     */
    int SearchBatchCate(uint32_t task_num, uint32_t batch_group_num, uint32_t max_topk, char *cpu_batch_data,
                        uint64_t *data_offset, uint64_t allocated_bytes, bool enable_rt);

    int SearchUnify(char *cpu_data, uint32_t *data_param, uint64_t *data_offset);

    /**
     * @brief copy vector data to gpu when real time
     * @param data: data start address of vector
     * @param data_size: bytes of vector
     */
    int CopyToGpu(const void *data, size_t data_size);

    /**
     * @brief reserve data in gpu
     * @param num: num of vector
     */
    int Reserve(size_t num);

    int ResetInMem();

    uint32_t GetDocNum();

    void SetNeutronManagerInterface(NeutronManagerInterface *neutron_manager_interface);

private:
    size_t feature_size_;
    CALCULATORTYPE calculator_type_;
    GpuKernelIndex *gpu_kernel_index;
    NeutronManagerInterface *neutron_manager_interface_;
};

} // namespace gpu
} // namespace neutron

#endif // INTERFACE_GPU_INDEX_SEARCHER_H_
#endif // ENABLE_GPU_IN_MERCURY_
