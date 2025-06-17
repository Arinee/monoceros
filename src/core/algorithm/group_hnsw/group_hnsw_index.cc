#include "group_hnsw_index.h"
#include <xmmintrin.h>
#include "src/core/utils/index_meta_helper.h"
#include "src/core/common/common.h"
// #include "src/core/algorithm/hdfs_file_wrapper.h"
#include "src/core/algorithm/query_info.h"

MERCURY_NAMESPACE_BEGIN(core);

const int64_t GroupHnswIndex::MAX_LEVEL;
const uint64_t GroupHnswIndex::MAX_SCALING_FACTOR;
const uint32_t GroupHnswIndex::MAX_NEIGHBOR_CNT;

GroupHnswIndex::GroupHnswIndex()
    : random_(19820604),
      ef_construction_(400),
      max_level_(5),
      scaling_factor_(30),
      upper_neighbor_cnt_(0),
      neighbor_cnt_(0),
      group_max_doc_num_(0),
      group_doc_num_(0),
      build_threshold_(1000)
{}

int GroupHnswIndex::Create(IndexParams& index_params) {

    Index::SetIndexParams(index_params);
    if (!IndexMetaHelper::parseFrom(index_params.getString(PARAM_DATA_TYPE),
                                    index_params.getString(PARAM_METHOD),
                                    index_params.getUint64(PARAM_DIMENSION),
                                    index_meta_)) {
        LOG_ERROR("Failed to init index meta.");
        return -1;
    }

    // 设置构建HNSW图阈值
    build_threshold_ = index_params.getUint32(PARAM_GROUP_HNSW_BUILD_THRESHOLD);
    if (build_threshold_ <= 0) {
        LOG_ERROR("mercury.group_hnsw.build_threshold must larger than 0");
        return -1;
    }
    
    // 设置HNSW图最大层数
    uint32_t max_level = index_params.getUint32(PARAM_HNSW_BUILDER_MAX_LEVEL);
    if (max_level <= 0) {
        LOG_ERROR("Max level for hnsw must > 0");
        return IndexError_InvalidArgument;
    }
    if (max_level > MAX_LEVEL) {
        LOG_WARN("Max level is too big, use max level [%ld]", MAX_LEVEL);
        max_level = MAX_LEVEL;
    }
    // _maxLevel从0开始
    max_level_ = max_level - 1;
    group_hnsw_info_manager_.SetMaxLevel(max_level_);

    // 设置缩放系数
    scaling_factor_ = index_params.getUint64(PARAM_HNSW_BUILDER_SCALING_FACTOR);
    if (scaling_factor_ == 0U) {
        LOG_WARN("Scaling factor for hnsw must > 0, have set scaling factor to: 30");
        scaling_factor_ = 30;
    } else if ( scaling_factor_ > MAX_SCALING_FACTOR) {
        LOG_WARN("Scaling factor is too big, use max level [%lu]", MAX_SCALING_FACTOR);
        scaling_factor_ = MAX_SCALING_FACTOR;
    }
    group_hnsw_info_manager_.SetScalingFactor(scaling_factor_);
    
    // 设置HNSW图中每个点的邻居数，第0层邻居数为其它层两倍
    upper_neighbor_cnt_ = index_params.getUint64(PARAM_HNSW_BUILDER_UPPER_NEIGHBOR_CNT);
    if (upper_neighbor_cnt_ == 0UL || upper_neighbor_cnt_ > MAX_NEIGHBOR_CNT) {
        LOG_ERROR("Value [%lu] for [%s] is invalid, it should between 0-%u", 
                  upper_neighbor_cnt_, PARAM_HNSW_BUILDER_UPPER_NEIGHBOR_CNT, 
                  MAX_NEIGHBOR_CNT);
        return IndexError_InvalidArgument;
    }
    neighbor_cnt_ = upper_neighbor_cnt_ * 2;
    group_hnsw_info_manager_.SetNeighborCnt(upper_neighbor_cnt_);

    //设置doc数量
    size_t max_build_num = index_params.getUint64(PARAM_GENERAL_MAX_BUILD_NUM);
    if (max_build_num > 0) {
        group_max_doc_num_ = max_build_num;
    } else {
        LOG_ERROR("Not set param: mercury.general.index.max_build_num");
        return -1;
    }

    coarse_hnsw_index_.setIndexMeta(&index_meta_, ContainFeature());

    //设置每层邻居在邻居信息中的偏移
    coarse_hnsw_index_.setLevelOffset(max_level_, upper_neighbor_cnt_);

    //设置每次计算与邻居点距离时的步长
    int step = index_params.getInt64(PARAM_GRAPH_COMMON_SEARCH_STEP);
    if (step <= 0) {
        LOG_WARN("Using 0 layer neighbor count as search step[%lu]", neighbor_cnt_);
        step = static_cast<int>(neighbor_cnt_);
    }
    coarse_hnsw_index_.setSearchStep(step);

    //设置候选点数每层topk
    ef_construction_ = index_params.getUint64(PARAM_HNSW_BUILDER_EFCONSTRUCTION);
    if (ef_construction_ == 0) {
        LOG_ERROR("[%s] must > 0", PARAM_HNSW_BUILDER_EFCONSTRUCTION);
        return IndexError_InvalidArgument;
    }
    coarse_hnsw_index_.setCandidateNums(max_level_, ef_construction_);

    //设置最大扫描数量
    int max_scan_num = index_params.getUint64(PARAM_GRAPH_COMMON_MAX_SCAN_NUM);
    if (max_scan_num <= 0) {
        LOG_WARN("Using ef_construction_ as max_scan_num[%lu]", ef_construction_);
        max_scan_num = static_cast<int>(ef_construction_);
    }
    coarse_hnsw_index_.setMaxScanNums(max_scan_num);

    std::string custom_distance_method = index_params.getString(PARAM_CUSTOM_DISTANCE_METHOD);
    part_dimension_ = index_params.getUint32(PARAM_CUSTOMED_PART_DIMENSION);
    if (part_dimension_ == 0) {
        coarse_hnsw_index_.setScorer(0, CustomMethods::Default);
    } else {
        if (custom_distance_method == "") {
            LOG_ERROR("have set part_dimension, but not set custom_distance_method");
            return -1;
        } else if (custom_distance_method == "Permutation") {
            coarse_hnsw_index_.setScorer(part_dimension_, CustomMethods::Permutation);
        } else if (custom_distance_method == "CtrCvr") {
            coarse_hnsw_index_.setScorer(part_dimension_, CustomMethods::CtrCvr);
        } else if (custom_distance_method == "Simple") {
            coarse_hnsw_index_.setScorer(part_dimension_, CustomMethods::Simple);
        } else if (custom_distance_method == "Mobius") {
            coarse_hnsw_index_.setScorer(part_dimension_, CustomMethods::Mobius);
        } else if (custom_distance_method == "RelCvr") {
            coarse_hnsw_index_.setScorer(part_dimension_, CustomMethods::RelCvr);
        }
    }
    
    return 0;
}

int GroupHnswIndex::CalCoarseIndexCapacity(const std::unordered_map<GroupInfo, uint32_t, GroupHnswHash, GroupHnswCmp>& group_meta) {

    //读取预处理得到的group元信息，初始化group_manager_及group_hnsw_info_manager_
    std::vector<GroupInfo> groups;
    std::vector<std::pair<GroupInfo, GroupHnswInfo>> group_hnsw_infos;

    int hnsw_cnt = 0;
    int brute_cnt = 0;

    for (auto& group_map_info : group_meta) {
        groups.emplace_back(group_map_info.first);
        uint32_t doc_total_num = group_map_info.second;
        bool is_build_hnsw = doc_total_num > build_threshold_ ? true : false;
        if (is_build_hnsw) {
            hnsw_cnt++;
        } else {
            brute_cnt++;
        }
        group_hnsw_infos.emplace_back(std::make_pair(group_map_info.first, GroupHnswInfo(doc_total_num, is_build_hnsw)));
    }

    LOG_INFO("hnsw_cnt: %d, brute_cnt: %d", hnsw_cnt, brute_cnt);

    if (group_manager_.Create(std::move(groups)) == -1) {
        LOG_ERROR("group_manager_ create failed. group num:%lu", groups.size());
        return -1;
    }
    if (group_hnsw_info_manager_.Create(std::move(group_hnsw_infos)) == -1) {
        LOG_ERROR("group_hnsw_info_manager_ create failed. group num:%lu", group_hnsw_infos.size());
        return -1;
    }

    //计算每个group在coarse_hnsw_index中的偏移以及coarse_hnsw_index的capacity
    group_hnsw_info_manager_.CalGroupOffset();

    //为coarse_hnsw_index_分配空间
    uint64_t capacity = group_hnsw_info_manager_.GetCoarseHnswIndexCapacity();
    uint32_t group_num = group_hnsw_info_manager_.GetGroupNum();

    coarse_hnsw_base_.assign(capacity, 0);
    if (!coarse_hnsw_index_.create(coarse_hnsw_base_.data(), capacity, group_num)) {
        return -1;
    }

    //初始化coarse_hnsw_index_中group元信息
    for (size_t i = 0; i < group_num; i++) {
        const GroupHnswInfo& group_hnsw_info = group_hnsw_info_manager_.GetGroupHnswInfo(i);
        if (coarse_hnsw_index_.setGroupMeta(group_hnsw_info.offset, group_hnsw_info.end_offset, 
                            group_hnsw_info.doc_total_num, group_hnsw_info.is_build_hnsw) != 0) {
            return -1;
        }
    }
    return 0;
}

int GroupHnswIndex::AssignSpace(docid_t doc_id, const std::vector<GroupInfo>& group_infos,
                                 std::vector<docid_t>& doc_ids,
                                 std::vector<uint32_t>& doc_max_layers,
                                 std::vector<uint32_t>& group_doc_ids,
                                 std::vector<uint64_t>& group_offsets,
                                 uint32_t doc_max_layer) 
{
    for (auto& group_info : group_infos) {
        uint64_t group_offset = group_hnsw_info_manager_.GetGroupOffset(group_info);

        // 非HNSW图直接写入group所在内存
        if(group_hnsw_info_manager_.IsHnswGroup(group_info) == false) {
            coarse_hnsw_index_.addBruteGroupDoc(group_offset, doc_id);
            continue;
        }

        // HNSW图元信息写入group所在内存
        int64_t doc_cur_num = coarse_hnsw_index_.addHnswGroupDocMeta(group_offset, doc_id, doc_max_layer);
        if (doc_cur_num == -1) {
            return -1;
        }
        doc_ids.push_back(doc_id);
        group_doc_ids.push_back(static_cast<uint32_t>(doc_cur_num));
        group_offsets.push_back(group_offset);
        doc_max_layers.push_back(doc_max_layer);
    }

    group_doc_num_++;
    return 0;
}

const void * GroupHnswIndex::GetDocFeature(docid_t doc_id) {
    return feature_profile_.getInfo(doc_id);
}

// docCnt[L + 1] / docCnt[L] == scaling_factor_
uint32_t GroupHnswIndex::GetRandomLevel()
{
    uint32_t level = 0;
    while (((random_() & 0xFFFFFFFF) < (0xFFFFFFFF / scaling_factor_)) && 
           (level < max_level_)) {
        level++;
    }
    return level;
}

int GroupHnswIndex::InitMappingSpace() {

    //为基类index中存储docid与pk之间映射关系的数据结构分配空间
    if (Index::Create(group_max_doc_num_) != 0) {
        LOG_ERROR("call index::create failed.");
        return -1;
    }

    // 为基类index中存储doc feature的数据结构分配存储空间
    auto elem_size = index_meta_.sizeofElement();
    auto feature_capacity = ArrayProfile::CalcSize(group_max_doc_num_, elem_size);
    feature_profile_base_.assign(feature_capacity, 0);
    if (!feature_profile_.create(feature_profile_base_.data(),
                                    feature_capacity, elem_size)) {
        LOG_ERROR("Failed to create feature profile");
        return -1;
    }

    //传入feature_profile_指针、index_meta_指针、创建scorer
    coarse_hnsw_index_.setFeatureProfile(&feature_profile_);
    
    return 0;
}

int GroupHnswIndex::BaseIndexAdd(docid_t doc_id, pk_t pk, const void *val, size_t len) 
{
    if (Index::Add(doc_id, pk, val, len)) {
        LOG_ERROR("failed to call Index::Add. doc_id: %u, pk: %lu, len: %lu", doc_id, pk, len);
        return -1;
    }

    if (!feature_profile_.insert(doc_id, val)) {
        LOG_ERROR("Failed to add into feature profile.");
        return -1;
    }

    return 0;
}

int GroupHnswIndex::AddDoc(docid_t group_doc_id, uint64_t group_offset, const void *val, uint32_t doc_max_layer) {
    if (coarse_hnsw_index_.addDoc(group_offset, group_doc_id, val, static_cast<int32_t>(doc_max_layer)) != 0) {
        return -1;
    }
    return 0;
}

void GroupHnswIndex::RedundantMemClip() {
    std::vector<uint64_t*> offsets;
    auto group_num = group_hnsw_info_manager_.GetGroupNum();
    offsets.reserve(group_num);
    for (uint32_t i = 0; i < group_num; i++) {
        uint64_t &offset = group_hnsw_info_manager_.GetGroupOffsetById(i);
        offsets.push_back(&offset);
    }
    coarse_hnsw_index_.RedundantMemClip(offsets);
    return;
}

int GroupHnswIndex::Dump(const void*& data, size_t& size) {
    
    if (DumpHelper::DumpCommon(this, index_package_) != 0) {
        LOG_ERROR("dump into package failed.");
        return -1;
    }

    index_package_.emplace(COMPONENT_DOC_NUM, &group_doc_num_, sizeof(group_doc_num_));

    index_package_.emplace(COMPONENT_GROUP_MANAGER, group_manager_.GetBaseStart(), group_manager_.GetCapacity());

    index_package_.emplace(COMPONENT_GROUP_HNSW_INFO_MANAGER, group_hnsw_info_manager_.GetBaseStart(), group_hnsw_info_manager_.GetCapacity());

    index_package_.emplace(COMPONENT_COARSE_HNSW_INDEX, coarse_hnsw_index_.GetBasePtr(), coarse_hnsw_index_.getHeader()->capacity);

    if (ContainFeature()) {
        index_package_.emplace(COMPONENT_FEATURE_PROFILE, feature_profile_.getHeader(), feature_profile_.getHeader()->capacity);
    }

    index_package_.emplace(COMPONENT_MAX_DOC_NUM, &group_max_doc_num_, sizeof(group_max_doc_num_));

    index_package_.emplace(COMPONENT_BUILD_THRESHOLD, &build_threshold_, sizeof(build_threshold_));

    index_package_.emplace(COMPONENT_MAXLEVEL, &max_level_, sizeof(max_level_));

    index_package_.emplace(COMPONENT_SCALING_FACTOR, &scaling_factor_, sizeof(scaling_factor_));

    index_package_.emplace(COMPONENT_UPPER_NEIGHBOR_CNT, &upper_neighbor_cnt_, sizeof(upper_neighbor_cnt_));

    index_package_.emplace(COMPONENT_EF_CONSTRUCTION, &ef_construction_, sizeof(ef_construction_));

    if (!index_package_.dump(data, size)) {
        LOG_ERROR("Failed to dump package.");
        return -1;
    }

    LOG_INFO("GroupHnswIndex::Dump End! index_package_ size: %lu", size);

    return 0;
}

void GroupHnswIndex::FreeMem() {
    coarse_hnsw_base_.clear();
    coarse_hnsw_base_.shrink_to_fit();
    feature_profile_base_.clear();
    feature_profile_base_.shrink_to_fit();
}

int GroupHnswIndex::Load(const void* data, size_t size) {
    
    LOG_INFO("-------Begin to Load group_hnsw_index. size: %lu.-------", size);

    if (Index::Load(data, size) != 0) {
        LOG_ERROR("Failed to call Index::Load.");
        return -1;
    }

    auto *component = index_package_.get(COMPONENT_DOC_NUM);
    if (!component) {
        LOG_ERROR("get component error: %s", COMPONENT_DOC_NUM);
        return -1;
    }
    group_doc_num_ = *((uint64_t*)component->getData());

    component = index_package_.get(COMPONENT_GROUP_MANAGER);
    if (!component) {
        LOG_ERROR("get component error: %s", COMPONENT_GROUP_MANAGER);
        return -1;
    }
    if (group_manager_.Load((void *)component->getData(), component->getDataSize()) != 0) {
        LOG_ERROR("group manager init error");
        return -1;
    }

    component = index_package_.get(COMPONENT_GROUP_HNSW_INFO_MANAGER);
    if (!component) {
        LOG_ERROR("get component error: %s", COMPONENT_GROUP_HNSW_INFO_MANAGER);
        return -1;
    }
    if (group_hnsw_info_manager_.Load((void *)component->getData(), component->getDataSize()) != 0) {
        LOG_ERROR("group_hnsw_info_manager init error");
        return -1;
    }

    component = index_package_.get(COMPONENT_COARSE_HNSW_INDEX);
    if (!component) {
        LOG_ERROR("get component error: %s", COMPONENT_COARSE_HNSW_INDEX);
        return -1;
    }
    if (coarse_hnsw_index_.load((void *)component->getData(), component->getDataSize()) != 0) {
        LOG_ERROR("coarse_hnsw_index_ init error");
        return -1;
    }

    coarse_hnsw_index_.setIndexMeta(&index_meta_, ContainFeature());

    if (ContainFeature()) {
        component = index_package_.get(COMPONENT_FEATURE_PROFILE);
        if (!component) {
            LOG_WARN("get component error: %s", COMPONENT_FEATURE_PROFILE);
        } else {
            if (!feature_profile_.load((void*)component->getData(), component->getDataSize())) {
                LOG_ERROR("feature profile load error");
                return -1;
            }
            coarse_hnsw_index_.setFeatureProfile(&feature_profile_);
        }
    }

    component = index_package_.get(COMPONENT_MAX_DOC_NUM);
    if (!component) {
        LOG_ERROR("get component error: %s", COMPONENT_MAX_DOC_NUM);
        return -1;
    }
    group_max_doc_num_ = *((uint64_t*)component->getData());

    component = index_package_.get(COMPONENT_BUILD_THRESHOLD);
    if (!component) {
        LOG_ERROR("get component error: %s", COMPONENT_BUILD_THRESHOLD);
        return -1;
    }
    build_threshold_ = *((uint32_t*)component->getData());

    component = index_package_.get(COMPONENT_MAXLEVEL);
    if (!component) {
        LOG_ERROR("get component error: %s", COMPONENT_MAXLEVEL);
        return -1;
    }
    max_level_ = *((uint32_t*)component->getData());

    component = index_package_.get(COMPONENT_SCALING_FACTOR);
    if (!component) {
        LOG_ERROR("get component error: %s", COMPONENT_SCALING_FACTOR);
        return -1;
    }
    scaling_factor_ = *((uint64_t*)component->getData());

    component = index_package_.get(COMPONENT_UPPER_NEIGHBOR_CNT);
    if (!component) {
        LOG_ERROR("get component error: %s", COMPONENT_UPPER_NEIGHBOR_CNT);
        return -1;
    }
    upper_neighbor_cnt_ = *((uint64_t*)component->getData());
    neighbor_cnt_ = upper_neighbor_cnt_ * 2;

    coarse_hnsw_index_.setLevelOffset(max_level_, upper_neighbor_cnt_);
    component = index_package_.get(COMPONENT_EF_CONSTRUCTION);
    if (!component) {
        LOG_ERROR("get component error: %s", COMPONENT_EF_CONSTRUCTION);
        return -1;
    }
    ef_construction_ = *((uint64_t*)component->getData());
    coarse_hnsw_index_.setCandidateNums(max_level_, ef_construction_);

    int step = index_params_.getInt64(PARAM_GRAPH_COMMON_SEARCH_STEP);
    if (step <= 0) {
        LOG_WARN("Using 0 layer neighbor count as search step[%lu]", neighbor_cnt_);
        step = static_cast<int>(neighbor_cnt_);
    }
    coarse_hnsw_index_.setSearchStep(step);

    int max_scan_num = index_params_.getUint64(PARAM_GRAPH_COMMON_MAX_SCAN_NUM);
    if (max_scan_num <= 0) {
        LOG_WARN("Using ef_construction_ as max_scan_num[%lu]", ef_construction_);
        max_scan_num = static_cast<int>(ef_construction_);
    }
    coarse_hnsw_index_.setMaxScanNums(max_scan_num);

    std::string custom_distance_method = index_params_.getString(PARAM_CUSTOM_DISTANCE_METHOD);
    part_dimension_ = index_params_.getUint32(PARAM_CUSTOMED_PART_DIMENSION);
    if (part_dimension_ == 0) {
        coarse_hnsw_index_.setScorer(0, CustomMethods::Default);
    } else {
        if (custom_distance_method == "") {
            LOG_ERROR("have set part_dimension, but not set custom_distance_method");
            return -1;
        } else if (custom_distance_method == "Permutation") {
            coarse_hnsw_index_.setScorer(part_dimension_, CustomMethods::Permutation);
        } else if (custom_distance_method == "CtrCvr") {
            coarse_hnsw_index_.setScorer(part_dimension_, CustomMethods::CtrCvr);
        } else if (custom_distance_method == "Simple") {
            coarse_hnsw_index_.setScorer(part_dimension_, CustomMethods::Simple);
        } else if (custom_distance_method == "Mobius") {
            coarse_hnsw_index_.setScorer(part_dimension_, CustomMethods::Mobius);
        } else if (custom_distance_method == "RelCvr") {
            coarse_hnsw_index_.setScorer(part_dimension_, CustomMethods::RelCvr);
        }
    }
    
    return 0;
}

int GroupHnswIndex::KnnSearch(GroupInfo group_info, size_t topk, const void * query_val, size_t len, GeneralSearchContext* context, MyHeap<DistNode>* group_heap, int max_scan_num_in_query, std::pair<int, int>& cmp_cnt)
{
    if (query_val == nullptr) {
        LOG_ERROR("Searching vector pointer is nullptr");
        return IndexError_InvalidArgument;
    }
    if (part_dimension_ == 0) {
        if (len != index_meta_.sizeofElement()) {
            LOG_ERROR("Searching vector size[%lu] mismatch vector size[%lu] in meta",
                    len, index_meta_.sizeofElement());
            return IndexError_Mismatch;
        }
    }
    
    uint64_t compare_cnt = 0;
    uint64_t group_offset = group_hnsw_info_manager_.GetGroupOffset(group_info);
    if (group_offset == 0) {
        LOG_WARN("No group info in this column!");
        return -1;
    }
    bool is_hnsw_group = group_hnsw_info_manager_.IsHnswGroup(group_info);

    if (is_hnsw_group) {
        uint32_t topk_search = topk < 100 ? 100 : topk;
        TopkHeap topk_heap(topk_search);
        if (likely(group_info.level == 0)) {
            coarse_hnsw_index_.searchZeroHnswNeighbors(group_offset, topk_heap, query_val, len, compare_cnt, max_scan_num_in_query);
        } else {
            coarse_hnsw_index_.searchHnswNeighbors(group_offset, topk_heap, query_val, len, context, compare_cnt, max_scan_num_in_query);
        }

        const docid_t *key_ptr = nullptr;
        const float *dist_ptr = nullptr;
        int size = 0;
        topk_heap.peep(key_ptr, dist_ptr, size);

        char* p_base = coarse_hnsw_index_.getBase();
        char* group_base = p_base + group_offset;
        uint64_t *doc_offset_base = reinterpret_cast<uint64_t *>(group_base + sizeof(CoarseHnswIndex::GroupHeader));

        for (int i = 0; i < size; i++) {
            docid_t global_doc_id = coarse_hnsw_index_.getGlobalDocId(group_base, doc_offset_base, key_ptr[i]);
            group_heap->push(std::move(DistNode(global_doc_id, dist_ptr[i])));
        }
        cmp_cnt.second = compare_cnt;
    } else {
        std::vector<DistNode> dist_nodes;
        coarse_hnsw_index_.searchBruteNeighbors(group_offset, dist_nodes, query_val, len, compare_cnt);
        for (size_t i = 0; i < dist_nodes.size(); ++i) {
            group_heap->push(dist_nodes[i]);
        }
        cmp_cnt.first = dist_nodes.size();
    }

    group_heap->sort();

    return 0;
}

int GroupHnswIndex::BruteSearch(GroupInfo group_info, size_t topk, const void * query_val, size_t len, GeneralSearchContext*context, std::vector<MyHeap<DistNode>>& group_heaps, std::pair<int, int>& cmp_cnt) {
    if (query_val == nullptr) {
        LOG_ERROR("Searching vector pointer is nullptr");
        return IndexError_InvalidArgument;
    }
    if (part_dimension_ == 0) {
        if (len != index_meta_.sizeofElement()) {
            LOG_ERROR("Searching vector size[%lu] mismatch vector size[%lu] in meta",
                    len, index_meta_.sizeofElement());
            return IndexError_Mismatch;
        }
    }
    
    uint64_t compare_cnt = 0;
    uint64_t group_offset = group_hnsw_info_manager_.GetGroupOffset(group_info);
    if (group_offset == 0) {
        LOG_WARN("No group info in this column!");
        return -1;
    }
    bool is_hnsw_group = group_hnsw_info_manager_.IsHnswGroup(group_info);
    std::vector<DistNode> dist_nodes;

    if (is_hnsw_group) {
        coarse_hnsw_index_.bruteSearchHnswNeighbors(group_offset, dist_nodes, query_val, len, compare_cnt);
    } else {
        coarse_hnsw_index_.searchBruteNeighbors(group_offset, dist_nodes, query_val, len, compare_cnt);
    }

    MyHeap<DistNode> result(topk);
    for (size_t i = 0; i < dist_nodes.size(); i++) {
        result.push(std::move(dist_nodes.at(i)));
    }
    result.sort();
    group_heaps.push_back(std::move(result));
    cmp_cnt.first = dist_nodes.size();

    return 0;
}

MERCURY_NAMESPACE_END(core);
