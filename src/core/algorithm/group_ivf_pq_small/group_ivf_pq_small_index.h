#ifndef __MERCURY_CORE_GROUP_IVF_PQ_SMALL_INDEX_H__
#define __MERCURY_CORE_GROUP_IVF_PQ_SMALL_INDEX_H__

#include "src/core/algorithm/centroid_resource.h"
#include "src/core/algorithm/centroid_resource_manager.h"
#include "src/core/algorithm/coarse_index.h"
#include "src/core/algorithm/flat_coarse_index.h"
#include "src/core/algorithm/group_manager.h"
#include "src/core/algorithm/index.h"
#include "src/core/algorithm/orig_dist_scorer.h"
#include <future>

MERCURY_NAMESPACE_BEGIN(core);

typedef std::unique_ptr<char[]> MatrixPointer;
typedef std::priority_queue<CentroidInfo, std::vector<CentroidInfo>, std::greater<CentroidInfo>> CentroidQueue;

std::string GetGroupDirName(const GroupInfo &group_info);
int16_t GetOrDefault(const IndexParams &params, const std::string &key, const uint16_t default_value);

class GroupIvfPqSmallIndex : public Index
{
public:
    typedef std::shared_ptr<GroupIvfPqSmallIndex> Pointer;
    GroupIvfPqSmallIndex();
    /// load index from index package file and index will save this handle
    int Load(const void *, size_t) override;
    /// add a new vectoV
    int Add(docid_t doc_id, pk_t pk, const std::string &query_str, const std::string &primary_key = "") override;
    /// Display the actual class name and some more info
    void Display() const override {};
    /// whither index segment full
    bool IsFull() const override;

    // rank score when generate reclaim map in merger
    float GetRankScore(docid_t doc_id);

    size_t MemQuota2DocCount(size_t memQuota, size_t elemSize) const override;

    size_t GetDocNum() const override
    {
        return group_doc_num_;
    }
    void SetDocNum(size_t group_doc_num)
    {
        group_doc_num_ = group_doc_num;
    }

    uint64_t GetMaxDocNum() const override
    {
        return group_max_doc_num_;
    }

    virtual int Create(IndexParams &index_params);

    int CopyInit(const Index *, size_t) override;

    int Dump(const void *&data, size_t &size) override;

    int64_t UsedMemoryInCurrent() const override;

    SlotIndex GetNearestGroupLabel(const void *data, size_t /* size */, gindex_t group_index,
                                   const CentroidResourceManager &centroid_resource_manager);

public:
    CoarseIndex<SmallBlock> &GetCoarseIndex()
    {
        return coarse_index_;
    }

    const CoarseIndex<SmallBlock> &GetCoarseIndex() const
    {
        return coarse_index_;
    }

    FlatCoarseIndex *GetFlatCoarseIndex()
    {
        return &flat_coarse_index_;
    }

    const FlatCoarseIndex *GetFlatCoarseIndex() const
    {
        return &flat_coarse_index_;
    }

    const ArrayProfile &GetFeatureProfile() const
    {
        return feature_profile_;
    }

    ArrayProfile &GetFeatureProfile()
    {
        return feature_profile_;
    }

    const GroupManager &GetGroupManager() const
    {
        return group_manager_;
    }

    const CentroidResourceManager &GetCentroidResourceManager() const
    {
        return centroid_resource_manager_;
    }

    const CentroidResourceManager &GetFineCentroidResourceManager() const
    {
        return fine_centroid_resource_manager_;
    }

    const CentroidResource &GetPqCentroidResource() const
    {
        return centroid_resource_manager_.GetCentroidResource(0);
    }
    CentroidResource &GetPqCentroidResource()
    {
        return centroid_resource_manager_.GetCentroidResource(0);
    }

    const ArrayProfile &GetPqCodeProfile() const
    {
        return pq_code_profile_;
    }
    ArrayProfile &GetPqCodeProfile()
    {
        return pq_code_profile_;
    }

    const ArrayProfile &GetRankScoreProfile() const
    {
        return rank_score_profile_;
    }
    ArrayProfile &GetRankScoreProfile()
    {
        return rank_score_profile_;
    }

    ArrayProfile &GetDocCreateTimeProfile()
    {
        return doc_create_time_profile_;
    }

    int AddBase(docid_t doc_id, pk_t pk)
    {
        return Index::Add(doc_id, pk, nullptr, 0);
    }

    int SearchIvf(gindex_t group_index, const void *datas, size_t size, size_t query_dimension,
                  const IndexParams &context_params, std::vector<off_t> &real_slot_indexs);

    void RecoverPostingFromSlot(std::vector<CoarseIndex<SmallBlock>::PostingIterator> &postings,
                                std::vector<uint32_t> &ivf_postings_group_ids,
                                std::vector<std::vector<off_t>> &real_slot_indexs,
                                std::vector<uint32_t> &group_doc_nums, bool is_multi_query);

    // for debug
    int SearchGroup(docid_t docid, std::vector<GroupInfo> &group_infos, std::vector<int> &labels) const;

protected:
    bool InitIvfCentroidMatrix(const std::string &centroid_dir);
    bool InitPqCentroidMatrix(const IndexParams &param);

    MatrixPointer DoLoadCentroidMatrix(const std::string &file_path, size_t dimension, size_t element_size,
                                       size_t &centroid_size, std::vector<uint16_t> *centroid_sizes,
                                       bool is_fine) const;
    bool StrToValue(const std::string &source, void *value) const;

private:
    struct BthreadMessage
    {
        GroupIvfPqSmallIndex *index;
        CentroidResource *centroid_resource;
        size_t start;
        size_t end;
        std::vector<CentroidInfo> *centroid_infos;
        const void *datas;
        size_t query_dimension;
    };

    static void *BthreadRun(void *message);

    int CalcCentroidScore(CentroidResource &centroid_resource, size_t start, size_t end,
                          std::vector<CentroidInfo> *centroid_infos, const void *datas, size_t query_dimension) const;

    int CalcFineCentroidScore(CentroidResource &centroid_resource, size_t start, size_t end,
                              std::vector<FineCentroidInfo> *centroid_infos, const void *datas, size_t query_dimension,
                              uint32_t coarse_index) const;

    int PushQueue(const std::vector<CentroidInfo> &infos, CentroidQueue &cq) const;

    bool NeedParralel(size_t centroid_num) const;

    size_t GetFeatureSize() const
    {
        auto elem_size = sizeof(float);
        if (index_meta_.type() == IndexMeta::kTypeHalfFloat) {
            elem_size = sizeof(half_float::half);
        }
        return elem_size * index_meta_.dimension();
    }

    size_t GetFragmentNum() const
    {
        return GetPqCentroidResource().getIntegrateMeta().fragmentNum;
    }

    size_t GetProductSize() const
    {
        return sizeof(uint16_t) * GetFragmentNum();
    }

    bool ResolveGroupFile(const std::string &meta_path, std::vector<GroupInfo> &groups,
                          std::unordered_map<GroupInfo, uint32_t, GroupHash, GroupCmp> &group_centroids);
    MatrixPointer NullMatrix() const;

    bool NeedTruncate(size_t posting_num, size_t group_index) const;

    uint64_t GetMaxCoarseDocNum() const
    {
        uint32_t max_level = index_params_.getUint32(PARAM_GENERAL_MAX_GROUP_LEVEL_NUM);
        if (max_level == 0) {
            max_level = DEFAULT_MAX_GROUP_LEVEL_NUM;
        }

        return GetMaxDocNum() * max_level;
    }

    uint64_t GetMaxCoarseDocNum(int32_t max_doc_num) const
    {
        uint32_t max_level = index_params_.getUint32(PARAM_GENERAL_MAX_GROUP_LEVEL_NUM);
        if (max_level == 0) {
            max_level = DEFAULT_MAX_GROUP_LEVEL_NUM;
        }

        return max_doc_num * max_level;
    }

private:
    /// 中心点管理类：每个类目的中心点信息 + 0类目还存pq分段中心点信息
    CentroidResourceManager centroid_resource_manager_;
    CentroidResourceManager fine_centroid_resource_manager_;
    GroupManager group_manager_;
    CoarseIndex<SmallBlock> coarse_index_; // 改为大block
    std::vector<char> coarse_base_;
    std::string rough_matrix_;
    std::string fine_rough_matrix_;

    // docid -> vector detail
    std::vector<uint8_t> feature_profile_base_;
    ArrayProfile feature_profile_;
    uint64_t group_doc_num_ = 0;
    uint64_t group_max_doc_num_ = 0;
    uint32_t sort_build_group_level_ = 0;
    OrigDistScorer::Factory dist_scorer_factory_;

    //-----------------support pq------------------------//
    // docid -> pq_code
    ArrayProfile pq_code_profile_;
    std::vector<char> pq_code_base_;
    std::string integrate_matrix_;
    //-------------------------------------------------//

    // docid -> rankscore
    ArrayProfile rank_score_profile_;
    std::vector<char> rank_score_base_;

    // docid -> doc create time
    ArrayProfile doc_create_time_profile_;
    std::vector<char> doc_create_time_profile_base_;

    IndexMeta index_meta_L2_; // to calculate centroid L2 distance

    // For fast doc iteration
    FlatCoarseIndex flat_coarse_index_;

public:
    static constexpr size_t MIN_BUILD_COUNT = 10000;
};

MERCURY_NAMESPACE_END(core);

#endif //__MERCURY_CORE_GROUP_IVF_PQ_SMALL_INDEX_H__
