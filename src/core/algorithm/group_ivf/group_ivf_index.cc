#include "group_ivf_index.h"
#include "src/core/algorithm/hdfs_file_wrapper.h"
#include "src/core/algorithm/query_info.h"
#include "src/core/algorithm/searcher.h"
#include "src/core/algorithm/thread_common.h"
#include "src/core/utils/index_meta_helper.h"
#include "src/core/utils/note_util.h"
#include "src/core/utils/string_util.h"

#include "bthread/bthread.h"

MERCURY_NAMESPACE_BEGIN(core);

using namespace fslib;
using namespace fslib::fs;

GroupIvfIndex::GroupIvfIndex()
{
    SetThreadEnv();
}

std::string GetGroupDirName(const GroupInfo &group_info)
{
    // [类目层级]-[类目ID]
    return std::to_string(group_info.level) + "-" + std::to_string(group_info.id);
}

MatrixPointer GroupIvfIndex::NullMatrix() const
{
    return MatrixPointer(nullptr);
}

int GroupIvfIndex::Load(const void *data, size_t size)
{
    LOG_INFO("Begin to Load group_ivf_index. size: %lu.", size);
    if (Index::Load(data, size) != 0) {
        LOG_ERROR("Failed to call Index::Load.");
        return -1;
    }

    auto *component = index_package_.get(COMPONENT_GROUP_MANAGER);
    if (!component) {
        LOG_ERROR("get component error: %s", COMPONENT_GROUP_MANAGER);
        return -1;
    }
    if (group_manager_.Load((void *)component->getData(), component->getDataSize()) != 0) {
        LOG_ERROR("group manager init error");
        return -1;
    }

    component = index_package_.get(COMPONENT_MAX_DOC_NUM);
    if (!component) {
        LOG_ERROR("get component error: %s", COMPONENT_MAX_DOC_NUM);
        return -1;
    }
    group_max_doc_num_ = *((uint64_t *)component->getData());

    sort_build_group_level_ = index_params_.getUint32(PARAM_SORT_BUILD_GROUP_LEVEL);
    LOG_INFO("sort build group level: %u", sort_build_group_level_);

    component = index_package_.get(COMPONENT_DOC_NUM);
    if (!component) {
        LOG_ERROR("get component error: %s", COMPONENT_DOC_NUM);
        return -1;
    }
    group_doc_num_ = *((uint64_t *)component->getData());

    component = index_package_.get(COMPONENT_ROUGH_MATRIX);
    if (!component) {
        LOG_ERROR("get component error: %s", COMPONENT_ROUGH_MATRIX);
        return -1;
    }
    if (centroid_resource_manager_.LoadRough((void *)component->getData(), component->getDataSize()) != 0) {
        LOG_ERROR("centroid resource manager init error");
        return -1;
    }
    if (EnableFineCluster()) {
        component = index_package_.get(COMPONENT_FINE_ROUGH_MATRIX);
        if (!component) {
            LOG_ERROR("get component error: %s", COMPONENT_FINE_ROUGH_MATRIX);
            return -1;
        }
        if (fine_centroid_resource_manager_.LoadRough((void *)component->getData(), component->getDataSize()) != 0) {
            LOG_ERROR("fine centroid resource manager init error");
            return -1;
        }
    }

    component = index_package_.get(COMPONENT_COARSE_INDEX);
    if (!component) {
        LOG_ERROR("get component error: %s", COMPONENT_COARSE_INDEX);
        return -1;
    }
    if (!coarse_index_.load((void *)component->getData(), component->getDataSize())) {
        LOG_ERROR("coarse index load error");
        return -1;
    }

    if (deletion_map_retriever_.isValid()) {
        LOG_INFO("genrate new coarse index after deletionmap");
        size_t slot_num = coarse_index_.getSlotNum(), used_doc_num = coarse_index_.getUsedDocNum();
        auto capacity = CoarseIndex<SmallBlock>::calcSize(slot_num, used_doc_num);
        coarse_base_.assign(capacity, 0);
        CoarseIndex<SmallBlock> new_coarse_index;
        if (!new_coarse_index.create(coarse_base_.data(), capacity, slot_num, used_doc_num)) {
            LOG_ERROR("Failed to create New CoarseIndex.");
            return -1;
        }

        for (size_t i = 0; i < slot_num; i++) {
            auto iter = coarse_index_.search(i);
            while (!iter.finish()) {
                uint32_t docid = iter.next();
                if (deletion_map_retriever_(base_docid_ + docid)) {
                    continue;
                }
                new_coarse_index.addDoc(i, docid);
            }
        }

        LOG_INFO("old coarse index docnum: %u, new coarse index docnum: %u", used_doc_num, new_coarse_index.getUsedDocNum());
        coarse_index_ = new_coarse_index;
    }

    if (is_gpu_) {
        flat_coarse_index_.ConvertFromCoarseIndex(coarse_index_);
        LOG_INFO("float coarse index converted, %u", coarse_index_.getUsedDocNum());
    }

    // 为了兼容老索引，后续下掉
    component = index_package_.get(COMPONENT_REDINDEX_DOCID_ARRAY);
    if (!component) {
        LOG_WARN("no REDINDEX_DOCID_ARRAY component.");
    } else {
        std::vector<int32_t> redindex_docids;
        auto ptr = static_cast<decltype(redindex_docids)::value_type *>(component->getData());
        redindex_docids.assign(ptr, ptr + component->getDataSize() / sizeof(decltype(redindex_docids)::value_type));
        coarse_index_.getHeader()->usedDocNum = redindex_docids.size();
    }

    if (ContainFeature()) {
        component = index_package_.get(COMPONENT_FEATURE_PROFILE);
        if (!component) {
            LOG_ERROR("get component error: %s", COMPONENT_FEATURE_PROFILE);
            return -1;
        } else {
            if (!feature_profile_.load((void *)component->getData(), component->getDataSize())) {
                LOG_ERROR("feature profile load error");
                return -1;
            }
        }
    }

    if (MultiAgeMode()) {
        LOG_INFO("multi age mode, load doc create time profile");
        component = index_package_.get(COMPONENT_DOC_CREATE_TIME_PROFILE);
        if (!component) {
            LOG_ERROR("get component error: %s", COMPONENT_DOC_CREATE_TIME_PROFILE);
            return -1;
        } else {
            if (!doc_create_time_profile_.load((void *)component->getData(), component->getDataSize())) {
                LOG_ERROR("doc create time profile load error");
                return -1;
            }
        }
    }

    component = index_package_.get(COMPONENT_RANK_SCORE_PROFILE);
    if (!component) {
        LOG_ERROR("get component error: %s", COMPONENT_RANK_SCORE_PROFILE);
        return -1;
    }
    if (!rank_score_profile_.load((void *)component->getData(), component->getDataSize())) {
        LOG_ERROR("rank score load error");
        return -1;
    }

    dist_scorer_factory_.Init(index_meta_);

    index_meta_L2_ = index_meta_;
    if (index_meta_.type() == mercury::core::IndexMeta::kTypeFloat) {
        index_meta_L2_.setMethod(mercury::core::IndexDistance::kMethodFloatSquaredEuclidean);
    } else if (index_meta_.type() == mercury::core::IndexMeta::kTypeHalfFloat) {
        index_meta_L2_.setMethod(mercury::core::IndexDistance::kMethodHalfFloatSquaredEuclidean);
    } else if (index_meta_.type() == mercury::core::IndexMeta::kTypeInt16) {
        index_meta_L2_.setMethod(mercury::core::IndexDistance::kMethodInt16SquaredEuclidean);
    } else if (index_meta_.type() == mercury::core::IndexMeta::kTypeInt8) {
        index_meta_L2_.setMethod(mercury::core::IndexDistance::kMethodInt8SquaredEuclidean);
    } else {
        LOG_ERROR("Not supported type[%d] for L2", index_meta_.type());
        return false;
    }

    return 0;
}

int GroupIvfIndex::CopyInit(const Index *index, size_t doc_num)
{
    if (!index) {
        LOG_ERROR("Invalid index pointer.");
        return -1;
    }

    if (Index::CopyInit(index, doc_num) != 0) {
        LOG_ERROR("Failed to call Index::CopyInit.");
        return -1;
    }

    group_max_doc_num_ = doc_num + index_params_.getUint64(PARAM_GENERAL_RESERVED_DOC);
    sort_build_group_level_ = index_params_.getUint32(PARAM_SORT_BUILD_GROUP_LEVEL);
    LOG_INFO("sort build group level: %u", sort_build_group_level_);

    const GroupIvfIndex *ivf_index = dynamic_cast<const GroupIvfIndex *>(index);
    centroid_resource_manager_ = ivf_index->GetCentroidResourceManager();
    if (EnableFineCluster()) {
        fine_centroid_resource_manager_ = ivf_index->GetFineCentroidResourceManager();
    }
    group_manager_ = ivf_index->GetGroupManager();

    auto slot_num = EnableFineCluster() ? fine_centroid_resource_manager_.GetTotalCentroidsNum()
                                        : centroid_resource_manager_.GetTotalCentroidsNum();
    auto capacity = CoarseIndex<SmallBlock>::calcSize(slot_num, GetMaxCoarseDocNum(group_max_doc_num_));
    coarse_base_.assign(capacity, 0);
    if (!coarse_index_.create(coarse_base_.data(), capacity, slot_num, GetMaxCoarseDocNum(group_max_doc_num_))) {
        LOG_ERROR("Failed to create CoarseIndex.");
        return -1;
    }

    if (ContainFeature()) {
        size_t elem_size = GetFeatureSize();
        size_t feature_elem_size = ForceHalf() ? elem_size / 2 : elem_size;
        capacity = ArrayProfile::CalcSize(group_max_doc_num_, feature_elem_size);
        feature_profile_base_.assign(capacity, 0);

        if (!feature_profile_.create(feature_profile_base_.data(), capacity, feature_elem_size)) {
            LOG_ERROR("Failed to create feature profile");
            return -1;
        }
    }

    if (MultiAgeMode()) {
        size_t elem_size = sizeof(uint32_t);
        auto capacity = ArrayProfile::CalcSize(group_max_doc_num_, elem_size);
        doc_create_time_profile_base_.assign(capacity, 0);
        if (!doc_create_time_profile_.create(doc_create_time_profile_base_.data(), capacity, elem_size)) {
            LOG_ERROR("Failed to create doc create time profile");
            return -1;
        }
    }

    auto elem_size = sizeof(float);
    if (index_meta_.type() == IndexMeta::kTypeHalfFloat) {
        elem_size = sizeof(half_float::half);
    }
    capacity = ArrayProfile::CalcSize(group_max_doc_num_, elem_size);
    rank_score_profile_base_.assign(capacity, 0);
    if (!rank_score_profile_.create(rank_score_profile_base_.data(), capacity, elem_size)) {
        LOG_ERROR("Failed to rank score profile");
        return -1;
    }

    return 0;
}

int GroupIvfIndex::Create(IndexParams &index_params)
{
    Index::SetIndexParams(index_params);
    if (!IndexMetaHelper::parseFrom(index_params.getString(PARAM_DATA_TYPE), index_params.getString(PARAM_METHOD),
                                    index_params.getUint64(PARAM_DIMENSION), index_meta_)) {
        LOG_ERROR("Failed to init ivf index meta.");
        return -1;
    }

    if (ForceHalf() && index_meta_.type() != IndexMeta::FeatureTypes::kTypeFloat) {
        LOG_ERROR("Force half only work when feature type is float, now feature type: %d", index_meta_.type());
        return -1;
    }

    if (!IndexMetaHelper::parseFrom(index_params.getString(PARAM_DATA_TYPE), "L2",
                                    index_params.getUint64(PARAM_DIMENSION), index_meta_L2_)) {
        LOG_ERROR("Failed to init ivf index meta L2.");
        return -1;
    }
    
    if (index_params.getBool(PARAM_MONOCEROS_IS_ONLINE)) {
        // 直接加载已有的中心点
        std::string centroid_path = index_params.getString(PARAM_MONOCEROS_ROOT_DIR) + "/mercury_index_centroid.package";
        std::string file_content;
        if (!HdfsFileWrapper::AtomicLoad(centroid_path, file_content)) {
            LOG_ERROR("load centroid file %s failed, reuse direct hdfs train path.", centroid_path.c_str());
        }
        else {
            if (!centroid_package_.load(file_content.data(), file_content.size())) {
                LOG_ERROR("Failed to load centroid pacakge");
                return -1;
            }
            auto *component = centroid_package_.get(COMPONENT_GROUP_MANAGER);
            if (!component) {
                LOG_ERROR("get component error: %s", COMPONENT_GROUP_MANAGER);
                return -1;
            }
            if (group_manager_.Load((void *)component->getData(),
                                    component->getDataSize()) != 0) {
                LOG_ERROR("group manager init error");
                return -1;
            }

            component = centroid_package_.get(COMPONENT_ROUGH_MATRIX);
            if (!component) {
                LOG_ERROR("get component error: %s", COMPONENT_ROUGH_MATRIX);
                return -1;
            }
            if (centroid_resource_manager_.LoadRough(
                    (void *)component->getData(), component->getDataSize()) != 0) {
                LOG_ERROR("centroid resource manager init error");
                return -1;
            }
            if (EnableFineCluster()) {
                component = centroid_package_.get(COMPONENT_FINE_ROUGH_MATRIX);
                if (!component) {
                    LOG_ERROR("get component error: %s", COMPONENT_FINE_ROUGH_MATRIX);
                    return -1;
                }
                if (fine_centroid_resource_manager_.LoadRough((void *)component->getData(), component->getDataSize()) != 0) {
                    LOG_ERROR("fine centroid resource manager init error");
                    return -1;
                }
            }
            LOG_INFO("Create Index using load centroid file %s success!", centroid_path.c_str());
        }
    }
    
    if  (centroid_package_.count() == 0) {
        if (index_params.getString(PARAM_TRAIN_DATA_PATH) != "") {
            if (index_params.getBool(PARAM_ENABLE_MIPS) &&
                !InitMipsTranform(index_params.getString(PARAM_TRAIN_DATA_PATH))) {
                LOG_ERROR("Failed to load mips transform");
                return -1;
            }

            if (!InitIvfCentroidMatrix(index_params.getString(PARAM_TRAIN_DATA_PATH))) {
                LOG_ERROR("Failed to load ivf centroid matrix");
                return -1;
            }
            LOG_INFO("Create Index using train data path file %s success!", index_params.getString(PARAM_TRAIN_DATA_PATH).c_str());
        } else {
            LOG_ERROR("Train path is empty.");
            return -1;
        }
    }

    sort_build_group_level_ = index_params.getUint32(PARAM_SORT_BUILD_GROUP_LEVEL);
    LOG_INFO("sort build group level: %u", sort_build_group_level_);

    auto elem_size = index_meta_.sizeofElement();
    size_t max_build_num = index_params.getUint64(PARAM_GENERAL_MAX_BUILD_NUM);
    if (max_build_num > 0) {
        group_max_doc_num_ = max_build_num;
    } else {
        auto mem_quota = index_params.getUint64(PARAM_GENERAL_INDEX_MEMORY_QUOTA);
        if (mem_quota == 0) {
            LOG_WARN("memory quota is not set, using default 1GB memory quota.");
            mem_quota = 1L * 1024L * 1024L * 1024L;
        }

        group_max_doc_num_ = MemQuota2DocCount(mem_quota, elem_size);
    }

    LOG_INFO("---------- Containable element count: %lu", group_max_doc_num_);
    if (group_max_doc_num_ == 0) {
        LOG_WARN("max doc num is 0, containFeature is %d", ContainFeature());
    }

    if (Index::Create(group_max_doc_num_) != 0) {
        LOG_ERROR("call index::create failed.");
        return -1;
    }

    // only support float now
    if (ContainFeature()) {
        auto feature_elem_size = ForceHalf() ? elem_size / 2 : elem_size;
        auto capacity = ArrayProfile::CalcSize(group_max_doc_num_, feature_elem_size);
        feature_profile_base_.assign(capacity, 0);

        if (!feature_profile_.create(feature_profile_base_.data(), capacity, feature_elem_size)) {
            LOG_ERROR("Failed to create feature profile");
            return -1;
        }
    }

    if (MultiAgeMode()) {
        elem_size = sizeof(uint32_t);
        auto capacity = ArrayProfile::CalcSize(group_max_doc_num_, elem_size);
        doc_create_time_profile_base_.assign(capacity, 0);
        if (!doc_create_time_profile_.create(doc_create_time_profile_base_.data(), capacity, elem_size)) {
            LOG_ERROR("Failed to create doc create time profile");
            return -1;
        }
    }

    // init CoarseIndex
    auto slot_num = EnableFineCluster() ? fine_centroid_resource_manager_.GetTotalCentroidsNum()
                                        : centroid_resource_manager_.GetTotalCentroidsNum();
    LOG_INFO("Total slot_num is %lu", slot_num);
    auto capacity = CoarseIndex<SmallBlock>::calcSize(slot_num, GetMaxCoarseDocNum());
    coarse_base_.assign(capacity, 0);
    if (!coarse_index_.create(coarse_base_.data(), capacity, slot_num, GetMaxCoarseDocNum())) {
        LOG_ERROR("Failed to create CoarseIndex.");
        return -1;
    }

    elem_size = sizeof(float);
    if (index_meta_.type() == IndexMeta::kTypeHalfFloat) {
        elem_size = sizeof(half_float::half);
    }
    capacity = ArrayProfile::CalcSize(group_max_doc_num_, elem_size);
    rank_score_profile_base_.assign(capacity, 0);
    if (!rank_score_profile_.create(rank_score_profile_base_.data(), capacity, elem_size)) {
        LOG_ERROR("Failed to create rank score profile");
        return -1;
    }

    dist_scorer_factory_.Init(index_meta_);

    return 0;
}

MatrixPointer GroupIvfIndex::DoLoadCentroidMatrix(const std::string &file_path, size_t dimension, size_t element_size,
                                                  size_t &centroid_size, std::vector<uint16_t> *centroid_sizes,
                                                  bool is_fine) const
{
    std::string file_content;
    if (!HdfsFileWrapper::AtomicLoad(file_path, file_content)) {
        LOG_ERROR("load centroid file failed.");
        return NullMatrix();
    }

    std::vector<std::string> centroid_vec = StringUtil::split(file_content, "\n");
    centroid_size = centroid_vec.size();
    char *centroid_matrix = new char[element_size * centroid_vec.size()];
    if (centroid_matrix == nullptr) {
        LOG_ERROR("centroid_matrix is null");
        return NullMatrix();
    }

    MatrixPointer matrix_pointer(centroid_matrix);

    // loop every centroid
    for (size_t i = 0; i < centroid_vec.size(); i++) {
        std::vector<std::string> centroid_item = StringUtil::split(centroid_vec.at(i), " ");
        if (centroid_item.size() != 2 + is_fine) {
            LOG_ERROR("centroid_item space split format error: %s", centroid_vec.at(i).c_str());
            return NullMatrix();
        }
        if (is_fine) {
            uint16_t centroid_index;
            if (!StringUtil::strToUInt16(centroid_item[0].c_str(), centroid_index)) {
                LOG_ERROR("centroid_index convert error: %s", centroid_item[0].c_str());
                return NullMatrix();
            }
            if (centroid_index < centroid_sizes->size()) {
                centroid_sizes->at(centroid_index)++;
            } else {
                LOG_ERROR("centroid_index is larger than centroid_sizes: %u, %lu", centroid_index,
                          centroid_sizes->size());
                return NullMatrix();
            }
        }

        std::vector<string> values = StringUtil::split(centroid_item.at(1 + is_fine), ",");
        if ((!EnableMips() && values.size() != dimension) || (EnableMips() && values.size() != dimension + 1)) {
            LOG_ERROR("centroid value commar split format error: %s", centroid_item.at(1 + is_fine).c_str());
            return NullMatrix();
        }
        // support binary vector
        if (index_meta_.type() == IndexMeta::FeatureTypes::kTypeBinary) {
            std::vector<char> bit_values;
            bit_values.assign(element_size, 0);
            // loop every dimension
            for (size_t j = 0; j < values.size(); j++) {
                float value;
                if (!StringUtil::strToFloat(values.at(j).c_str(), value)) {
                    LOG_ERROR("centroid value str to float format error: %s", values.at(j).c_str());
                    return NullMatrix();
                }
                uint32_t pos = j % 8;
                if (value <= 0.5) {
                    continue;
                } else if (value > 0.5 && value <= 1.0) {
                    char mask = 0x1 << (7u - pos);
                    char &value = bit_values.at(j / 8);
                    value |= mask;
                } else {
                    LOG_ERROR("invalid centroid vector value: %s, larger than 1.0, is not two-value",
                              values.at(j).c_str());
                    return NullMatrix();
                }
            }
            memcpy((char *)centroid_matrix + element_size * i, bit_values.data(), element_size);
        } else {
            // loop every dimension
            for (size_t j = 0; j < values.size(); j++) {
                char value[32]; // here at most int64_t 8 Byte, 32 is enough
                if (!StrToValue(values.at(j), (void *)value)) {
                    LOG_ERROR("centroid value str to value format error: %s", values.at(j).c_str());
                    return NullMatrix();
                }

                size_t type_size = element_size / dimension;
                memcpy((char *)centroid_matrix + element_size * i + j * type_size, (void *)value, type_size);
            }
        }
    }

    return std::move(matrix_pointer);
}

bool GroupIvfIndex::StrToValue(const std::string &source, void *value) const
{
    switch (index_meta_.type()) {
    case IndexMeta::FeatureTypes::kTypeUnknown:
        return false;
    case IndexMeta::FeatureTypes::kTypeBinary:
        // TODO, how to deal with binary

        return false;
    case IndexMeta::FeatureTypes::kTypeHalfFloat:
        if (!StringUtil::strToHalf(source.c_str(), *(half_float::half *)value)) {
            return false;
        }
        return true;
    case IndexMeta::FeatureTypes::kTypeFloat:
        if (!StringUtil::strToFloat(source.c_str(), *(float *)value)) {
            return false;
        }
        return true;
    case IndexMeta::FeatureTypes::kTypeDouble:
        if (!StringUtil::strToDouble(source.c_str(), *(double *)value)) {
            return false;
        }
        return true;
    case IndexMeta::FeatureTypes::kTypeInt8:
        if (!StringUtil::strToInt8(source.c_str(), *(int8_t *)value)) {
            return false;
        }
        return true;
    case IndexMeta::FeatureTypes::kTypeInt16:
        if (!StringUtil::strToInt16(source.c_str(), *(int16_t *)value)) {
            return false;
        }
        return true;
    }

    return false;
}

bool GroupIvfIndex::InitIvfCentroidMatrix(const std::string &centroid_dir)
{
    std::string rough_meta_path = centroid_dir + INDEX_META_FILE;
    std::vector<GroupInfo> groups;
    std::unordered_map<GroupInfo, uint32_t, GroupHash, GroupCmp> group_centroids;
    if (ResolveGroupFile(rough_meta_path, groups, group_centroids) == false) {
        LOG_ERROR("resolve group file failed.");
        return false;
    }

    if (group_manager_.Create(std::move(groups)) == -1) {
        LOG_ERROR("group_manager_ create failed. group num:%lu", groups.size());
        return -1;
    }

    for (gindex_t i = 0; i < group_manager_.GetGroupNum(); i++) {
        const GroupInfo &group_info = group_manager_.GetGroupInfo(i);
        auto iter = group_centroids.find(group_info);
        if (iter == group_centroids.end()) {
            LOG_ERROR("do not find group info");
            return false;
        }

        uint32_t centroids_num = iter->second;
        assert(centroids_num >= 1);
        size_t centroid_size = 0;
        MatrixPointer matrix_pointer;
        if (centroids_num > 1) { // 需要读取中心点文件
            std::string centroid_path = centroid_dir + "/" + GetGroupDirName(group_info) + IVF_CENTROID_FILE_POSTFIX;
            matrix_pointer =
                DoLoadCentroidMatrix(centroid_path, index_meta_.dimension(), index_meta_.sizeofElement(EnableMips()),
                                     centroid_size, nullptr, false);
            if (!matrix_pointer) {
                LOG_ERROR("Failed to Load centroid Matrix.");
                return false;
            }
        } else { // 只有一个中心点,mock一个
            centroid_size = 1;
            matrix_pointer.reset(new char[index_meta_.sizeofElement(EnableMips())]);
            memset(matrix_pointer.get(), 0, index_meta_.sizeofElement(EnableMips()));
        }

        char *centroid_matrix = matrix_pointer.get();
        std::vector<uint32_t> centroids_levelcnts;
        centroids_levelcnts.push_back(centroid_size);

        CentroidResource centroid_resource;
        // only support one level cnt, DefaultLevelCnt = 1.
        CentroidResource::RoughMeta rough_meta(index_meta_.sizeofElement(EnableMips()), DefaultLevelCnt,
                                               centroids_levelcnts);
        if (!centroid_resource.create(rough_meta)) {
            LOG_ERROR("Failed to create centroid resource.");
            return false;
        }

        // only 1 level
        for (size_t i = 0; i < centroid_size; i++) {
            if (!centroid_resource.setValueInRoughMatrix(
                    0, i, centroid_matrix + i * index_meta_.sizeofElement(EnableMips()))) {
                LOG_ERROR("Failed to set centroid resource rough matrix.");
                return false;
            }
        }

        centroid_resource_manager_.AddCentroidResource(std::move(centroid_resource));

        // fine cluster, 2nd level
        if (EnableFineCluster()) {
            std::vector<uint16_t> fine_centroid_sizes(centroid_size, 0);
            std::string centroid_path =
                centroid_dir + "/" + GetGroupDirName(group_info) + IVF_FINE_CENTROID_FILE_POSTFIX;
            size_t fine_centroid_size = 0;
            auto fine_matrix_pointer =
                DoLoadCentroidMatrix(centroid_path, index_meta_.dimension(), index_meta_.sizeofElement(EnableMips()),
                                     fine_centroid_size, &fine_centroid_sizes, true);
            if (!fine_matrix_pointer) {
                LOG_ERROR("Failed to Load fine centroid Matrix.");
                return false;
            }

            size_t prev_centriod_sum = 0;
            for (size_t j = 0; j < centroid_size; j++) {
                CentroidResource fine_centroid_resource;
                CentroidResource::RoughMeta fine_rough_meta(index_meta_.sizeofElement(EnableMips()), DefaultLevelCnt,
                                                            { fine_centroid_sizes[j] });
                if (!fine_centroid_resource.create(fine_rough_meta)) {
                    LOG_ERROR("Failed to create fine centroid resource.");
                    return false;
                }

                for (size_t k = 0; k < fine_centroid_sizes[j]; k++) {
                    if (!fine_centroid_resource.setValueInRoughMatrix(
                            0, k,
                            fine_matrix_pointer.get() +
                                (prev_centriod_sum + k) * index_meta_.sizeofElement(EnableMips()))) {
                        LOG_ERROR("Failed to set fine centroid resource rough matrix.");
                        return false;
                    }
                }
                fine_centroid_resource_manager_.AddCentroidResource(std::move(fine_centroid_resource));
                prev_centriod_sum += fine_centroid_sizes[j];
            }
        }
    }

    return true;
}

bool GroupIvfIndex::ResolveGroupFile(const std::string &meta_path, std::vector<GroupInfo> &groups,
                                     std::unordered_map<GroupInfo, uint32_t, GroupHash, GroupCmp> &group_centroids)
{
    // TODO, resolve group
    std::string file_content;
    if (!HdfsFileWrapper::AtomicLoad(meta_path, file_content)) {
        LOG_ERROR("load meta file failed.");
        return false;
    }

    std::vector<std::string> line_vec = StringUtil::split(file_content, "\n");
    for (size_t i = 0; i < line_vec.size(); i++) {
        std::vector<std::string> meta_vec = StringUtil::split(line_vec.at(i), ":");
        if (meta_vec.size() != 4) {
            LOG_ERROR("meta vec format error. meta: %s", line_vec.at(i).c_str());
            return false;
        }

        GroupInfo info;
        if (!StringUtil::strToUInt32(meta_vec.at(0).c_str(), info.level)) {
            LOG_ERROR("convert level failed. level:%s", meta_vec.at(0).c_str());
            return false;
        }

        if (!StringUtil::strToUInt32(meta_vec.at(1).c_str(), info.id)) {
            LOG_ERROR("convert id failed. level:%s", meta_vec.at(1).c_str());
            return false;
        }

        uint32_t centroid_num;
        if (!StringUtil::strToUInt32(meta_vec.at(3).c_str(), centroid_num)) {
            LOG_ERROR("convert centroid failed. centroid num:%s", meta_vec.at(3).c_str());
            return false;
        }

        group_centroids.insert({ info, centroid_num });
        groups.push_back(info);
    }

    return true;
}

bool GroupIvfIndex::InitMipsTranform(const std::string &centroid_dir)
{
    std::string mips_meta_path = centroid_dir + MIPS_META_FILE;
    std::string file_content;
    if (!HdfsFileWrapper::AtomicLoad(mips_meta_path, file_content)) {
        LOG_ERROR("load mips meta file failed.");
        return false;
    }
    if (!StrToValue(file_content, (void *)&mips_norm_)) {
        LOG_ERROR("load mips meta file failed.");
        return false;
    }
    LOG_INFO("load mips meta successful, %s", file_content.c_str());

    return true;
}

/// add a new vector
int GroupIvfIndex::Add(docid_t doc_id, pk_t pk, const std::string &query_str, const std::string &primary_key)
{

    QueryInfo query_info(query_str);
    if (index_meta_.type() == IndexMeta::FeatureTypes::kTypeBinary) {
        query_info.SetFeatureTypes(IndexMeta::FeatureTypes::kTypeBinary);
    }

    if (index_meta_.type() == IndexMeta::FeatureTypes::kTypeHalfFloat) {
        query_info.SetFeatureTypes(IndexMeta::FeatureTypes::kTypeHalfFloat);
    }

    if (!query_info.MakeAsBuilder()) {
        LOG_ERROR("resolve query failed. query str:%s", query_info.GetRawQuery().c_str());
        return -1;
    }

    if (MultiAgeMode()) {
        uint32_t create_timestamp_s = 0;
        if (unlikely(!NoteUtil::NoteIdToCreateTimeS(primary_key, create_timestamp_s))) {
            LOG_ERROR("parse note create timestamp failed, note id[%s], fill 0", primary_key.c_str());
        }
        doc_create_time_profile_.insert(doc_id, &create_timestamp_s);
    }

    const std::vector<GroupInfo> &group_infos = query_info.GetGroupInfos();
    if (Index::Add(doc_id, pk, query_info.GetVector(), query_info.GetVectorLen())) {
        LOG_ERROR("failed to call Index::Add. doc_id: %u, pk: %lu, len: %lu", doc_id, pk, query_info.GetVectorLen());
        return -1;
    }

    if ((size_t)doc_id >= GetMaxDocNum()) {
        LOG_ERROR("docid should be less than GetMaxDocNum");
        return -1;
    }

    if (ContainFeature() && ForceHalf()) {
        std::vector<half_float::half> half_query_vector(query_info.GetDimension());
        for (size_t i = 0; i < query_info.GetDimension(); i++) {
            half_query_vector[i] = *((float *)(query_info.GetVector()) + i);
        }
        if (!feature_profile_.insert(doc_id, half_query_vector.data())) {
            LOG_ERROR("Failed to add into feature profile.");
            return -1;
        }
    } else if (ContainFeature() && !feature_profile_.insert(doc_id, query_info.GetVector())) {
        LOG_ERROR("Failed to add into feature profile.");
        return -1;
    }

    if (group_infos.size() <= 0) {
        LOG_ERROR("query at least has one group");
        return -1;
    }

    if (searcher_ && !searcher_->Add(doc_id, query_info.GetVector())) {
        LOG_ERROR("Add Rt Neutron Index failed");
        return -1;
    }

    float score = 0;
    for (size_t i = 0; i < group_infos.size(); i++) {
        gindex_t group_index = group_manager_.GetGroupIndex(group_infos.at(i));
        if (group_index == INVALID_GROUP_INDEX) {
            LOG_ERROR("group info not in index. level: %u, id: %u", group_infos.at(i).level, group_infos.at(i).id);
            continue;
        }

        // 一层聚类
        SlotIndex label = GetNearestGroupLabel(query_info.GetVector(), query_info.GetVectorLen(), group_index,
                                               centroid_resource_manager_);
        if (label == INVALID_SLOT_INDEX) {
            LOG_ERROR("Failed to get nearest label.");
            return -1;
        }
        // 二层聚类
        if (EnableFineCluster()) {
            label = GetNearestGroupLabel(query_info.GetVector(), query_info.GetVectorLen(), label,
                                         fine_centroid_resource_manager_);
            if (label == INVALID_SLOT_INDEX) {
                LOG_ERROR("Failed to get nearest fine label.");
                return -1;
            }
        }

        if (!coarse_index_.addDoc(label, doc_id)) {
            LOG_ERROR("Failed to add into coarse index.");
            return -1;
        }

        if (group_infos.at(i).level == sort_build_group_level_) {
            score = static_cast<float>(log(label + 1));
        }
    }
    if (!rank_score_profile_.insert(doc_id, &score)) {
        LOG_ERROR("Failed to add into rank score.");
        return -1;
    }

    // single thread
    group_doc_num_++;
    return 0;
}

SlotIndex GroupIvfIndex::GetNearestGroupLabel(const void *data, size_t /* size */, gindex_t group_index,
                                              const CentroidResourceManager &centroid_resource_manager)
{
    SlotIndex return_labels = INVALID_SLOT_INDEX;
    size_t level = 0;
    const CentroidResource &centroid_resource = centroid_resource_manager.GetCentroidResource(group_index);
    if (!centroid_resource.IsInit()) {
        LOG_ERROR("centroid resource is not init yet.");
        return INVALID_SLOT_INDEX;
    }

    auto &roughMeta = centroid_resource.getRoughMeta();
    std::vector<size_t> levelScanLimit;
    for (size_t i = 0; i < roughMeta.levelCnt - 1; ++i) {
        levelScanLimit.push_back(roughMeta.centroidNums[i] / 10);
    }

    std::priority_queue<CentroidInfo, std::vector<CentroidInfo>, std::greater<CentroidInfo>> centroids;

    for (uint32_t i = 0; roughMeta.centroidNums.size() && i < roughMeta.centroidNums[level]; ++i) {
        const void *centroidValue = centroid_resource.getValueInRoughMatrix(level, i);
        float score = index_meta_L2_.distance(data, centroidValue);
        if (EnableMips()) {
            score += index_meta_L2_.getMipsScoreOffline(data, centroidValue, &mips_norm_);
        }

        centroids.emplace(i, score);
    }

    for (level = 1; level < roughMeta.levelCnt; ++level) {
        uint32_t centroidNum = roughMeta.centroidNums[level];
        decltype(centroids) candidate;
        candidate.swap(centroids);

        size_t scanNum = levelScanLimit[level - 1];
        while (!candidate.empty() && scanNum-- > 0) {
            auto doc = candidate.top();
            candidate.pop();
            for (uint32_t i = 0; i < centroidNum; ++i) {
                uint32_t centroid = doc.index * centroidNum + i;
                const void *centroidValue = centroid_resource.getValueInRoughMatrix(level, centroid);
                float dist = index_meta_L2_.distance(data, centroidValue);
                if (EnableMips()) {
                    dist += index_meta_L2_.getMipsScoreOffline(data, centroidValue, &mips_norm_);
                }

                centroids.emplace(centroid, dist);
            }
        }
    }

    if (!centroids.empty()) {
        return_labels = centroids.top().index;
    }

    return centroid_resource_manager.GetSlotIndex(group_index, return_labels);
}

/// whither index segment full
bool GroupIvfIndex::IsFull() const
{
    // here current doc num从coarseIndex里面拿
    return GetDocNum() >= GetMaxDocNum();
}

size_t GroupIvfIndex::MemQuota2DocCount(size_t memQuota, size_t /* elemSize */) const
{
    int64_t slotNum = EnableFineCluster() ? fine_centroid_resource_manager_.GetTotalCentroidsNum()
                                          : centroid_resource_manager_.GetTotalCentroidsNum();
    size_t elemCount = 0;
    size_t realMemUsed = 0;
    do {
        elemCount += MIN_BUILD_COUNT;
        realMemUsed = 0;
        realMemUsed += CoarseIndex<SmallBlock>::calcSize(slotNum, elemCount);
        if (ContainFeature()) {
            realMemUsed += ArrayProfile::CalcSize(elemCount, GetFeatureSize());
        }
    } while (realMemUsed <= memQuota);
    elemCount -= MIN_BUILD_COUNT;
    if (elemCount == 0) {
        std::cerr << "contain feature: " << ContainFeature() << std::endl;
        std::cerr << "memQuota: " << memQuota << ", elemCount: " << elemCount << std::endl;
        std::cerr << "min build count:" << MIN_BUILD_COUNT << std::endl;
        std::cerr << "slotNum:" << slotNum << ", feature size: " << GetFeatureSize() << std::endl;
        std::cerr << "coarse cap:" << CoarseIndex<SmallBlock>::calcSize(slotNum, MIN_BUILD_COUNT) << std::endl;
        std::cerr << "feature cap:" << ArrayProfile::CalcSize(MIN_BUILD_COUNT, GetFeatureSize()) << std::endl;
    }

    return elemCount;
}

int GroupIvfIndex::Dump(const void *&data, size_t &size)
{
    if (DumpHelper::DumpCommon(this, index_package_) != 0) {
        LOG_ERROR("dump into package failed.");
        return -1;
    }

    centroid_resource_manager_.DumpRoughMatrix(rough_matrix_);
    index_package_.emplace(COMPONENT_ROUGH_MATRIX, rough_matrix_.data(), rough_matrix_.size());

    if (EnableFineCluster()) {
        fine_centroid_resource_manager_.DumpRoughMatrix(fine_rough_matrix_);
        index_package_.emplace(COMPONENT_FINE_ROUGH_MATRIX, fine_rough_matrix_.data(), fine_rough_matrix_.size());
    }

    index_package_.emplace(COMPONENT_MAX_DOC_NUM, &group_max_doc_num_, sizeof(group_max_doc_num_));
    index_package_.emplace(COMPONENT_DOC_NUM, &group_doc_num_, sizeof(group_doc_num_));

    index_package_.emplace(COMPONENT_COARSE_INDEX, coarse_index_.GetBasePtr(), coarse_index_.getHeader()->capacity);
    if (ContainFeature()) {
        index_package_.emplace(COMPONENT_FEATURE_PROFILE, feature_profile_.getHeader(),
                               feature_profile_.getHeader()->capacity);
    }
    if (MultiAgeMode()) {
        LOG_INFO("multi age mode, dump doc create time profile");
        index_package_.emplace(COMPONENT_DOC_CREATE_TIME_PROFILE, doc_create_time_profile_.getHeader(),
                               doc_create_time_profile_.getHeader()->capacity);
    }
    index_package_.emplace(COMPONENT_GROUP_MANAGER, group_manager_.GetBaseStart(), group_manager_.GetCapacity());

    index_package_.emplace(COMPONENT_RANK_SCORE_PROFILE, rank_score_profile_.getHeader(),
                           rank_score_profile_.getHeader()->capacity);
    if (!index_package_.dump(data, size)) {
        LOG_ERROR("Failed to dump package.");
        return -1;
    }

    return 0;
}

int GroupIvfIndex::DumpCentroid(const void *&data, size_t &size)
{
    if (rough_matrix_.size() == 0) {
        LOG_ERROR("rough_matrix_ is empty");
        return -1;
    }
    centroid_package_.emplace(COMPONENT_ROUGH_MATRIX, rough_matrix_.data(),
                              rough_matrix_.size());

    centroid_package_.emplace(COMPONENT_GROUP_MANAGER,
                              group_manager_.GetBaseStart(),
                              group_manager_.GetCapacity());
    
    if (EnableFineCluster()) {
        fine_centroid_resource_manager_.DumpRoughMatrix(fine_rough_matrix_);
        centroid_package_.emplace(COMPONENT_FINE_ROUGH_MATRIX, fine_rough_matrix_.data(), fine_rough_matrix_.size());
    }

    if (!centroid_package_.dump(data, size)) {
        LOG_ERROR("Failed to dump centroid package.");
        return -1;
    }

    return 0;
}

int64_t GroupIvfIndex::UsedMemoryInCurrent() const
{
    int64_t result = sizeof(*this) + coarse_index_.getHeader()->usedSize + rank_score_profile_.getHeader()->capacity;
    if (ContainFeature()) {
        result += feature_profile_.getHeader()->capacity;
    }
    if (MultiAgeMode()) {
        result += doc_create_time_profile_.getHeader()->capacity;
    }

    return result;
}

bool GroupIvfIndex::NeedParralel(size_t centroid_num) const
{
    if (!mercury_need_parallel || centroid_num <= (size_t)mercury_min_parralel_centroids) {
        return false;
    }

    return true;
}

int GroupIvfIndex::PushQueue(const std::vector<CentroidInfo> &infos, CentroidQueue &cq) const
{
    for (const auto &info : infos) {
        cq.push(std::move(info));
    }

    return 0;
}

float GroupIvfIndex::BestDistance(const std::vector<const void *> &datas, size_t query_dimension,
                                  const void *centroid_value, OrigDistScorer scorer) const
{
    return scorer.Score(centroid_value, datas);
}

void *GroupIvfIndex::BthreadRun(void *message)
{
    BthreadMessage *msg = static_cast<BthreadMessage *>(message);
    if (msg->index) {
        msg->index->CalcCentroidScore(*msg->centroid_resource, msg->start, msg->end, msg->centroid_infos, msg->datas,
                                      msg->query_dimension);
    }
    return nullptr;
}

int GroupIvfIndex::CalcCentroidScore(CentroidResource &centroid_resource, size_t start, size_t end,
                                     std::vector<CentroidInfo> *centroid_infos, const void *datas,
                                     size_t query_dimension) const
{
    // OrigDistScorer scorer = dist_scorer_factory_.Create();
    const size_t first_level = 0;
    for (uint32_t i = start; i < end; i++) {
        float score = 0.0;
        const void *centroid_value = centroid_resource.getValueInRoughMatrix(first_level, i);
        if (index_meta_.type() == IndexMeta::FeatureTypes::kTypeBinary) {
            score = index_meta_.distance(datas, centroid_value);
        } else {
            score = index_meta_L2_.distance(datas, centroid_value);
            if (EnableMips()) {
                score += index_meta_L2_.getMipsScoreOnline(const_cast<void *>(centroid_value));
            }
            // score = BestDistance(datas, query_dimension, centroid_value, scorer);
        }
        centroid_infos->emplace_back(i, score);
    }
    return 0;
}

int GroupIvfIndex::CalcFineCentroidScore(CentroidResource &centroid_resource, size_t start, size_t end,
                                         std::vector<FineCentroidInfo> *centroid_infos, const void *datas,
                                         size_t query_dimension, uint32_t coarse_index) const
{
    // OrigDistScorer scorer = dist_scorer_factory_.Create();
    const size_t first_level = 0;
    for (uint32_t i = start; i < end; i++) {
        float score = 0.0;
        const void *centroid_value = centroid_resource.getValueInRoughMatrix(first_level, i);
        if (index_meta_.type() == IndexMeta::FeatureTypes::kTypeBinary) {
            score = index_meta_.distance(datas, centroid_value);
        } else {
            score = index_meta_L2_.distance(datas, centroid_value);
            if (EnableMips()) {
                score += index_meta_L2_.getMipsScoreOnline(const_cast<void *>(centroid_value));
            }
            // score = BestDistance(datas, query_dimension, centroid_value, scorer);
        }
        centroid_infos->emplace_back(i, score, coarse_index);
    }
    return 0;
}

int GroupIvfIndex::SearchIvf(gindex_t group_index, const void *datas, size_t size, size_t query_dimension,
                             const IndexParams &context_params, std::vector<off_t> &group_real_slot_indexs, bool is_rt,
                             bool is_multi_age)
{
    const IndexParams &index_params = GetIndexParams();
    CentroidResource &centroid_resource = centroid_resource_manager_.GetCentroidResource(group_index);

    float ratio = context_params.has(PARAM_COARSE_SCAN_RATIO) ? context_params.getFloat(PARAM_COARSE_SCAN_RATIO)
                                                              : index_params.getFloat(PARAM_COARSE_SCAN_RATIO);
    if (is_rt && is_multi_age && index_params.has(PARAM_RT_COARSE_SCAN_RATIO)) {
        ratio = context_params.has(PARAM_RT_COARSE_SCAN_RATIO) ? context_params.getFloat(PARAM_RT_COARSE_SCAN_RATIO)
                                                               : index_params.getFloat(PARAM_RT_COARSE_SCAN_RATIO);
    }
    float downgrade_percent =
        context_params.has(PARAM_DOWNGRADE_PERCENT) ? context_params.getFloat(PARAM_DOWNGRADE_PERCENT) : 1;
    size_t nprobe = static_cast<size_t>(ratio * downgrade_percent * centroid_resource.getLeafCentroidNum());
    if (nprobe < 1) {
        nprobe = 1;
    }

    auto &rough_meta = centroid_resource.getRoughMeta();
    auto &centroid_nums = rough_meta.centroidNums;

    std::priority_queue<CentroidInfo, std::vector<CentroidInfo>, std::greater<CentroidInfo>> centroid_resource_s;

    if (centroid_nums.empty()) {
        LOG_ERROR("centroid_nums is empty");
        return -1;
    }

    int first_level = 0;
    if (!NeedParralel(centroid_nums[first_level])) {
        std::vector<CentroidInfo> centroid_infos;
        CalcCentroidScore(centroid_resource, 0, centroid_nums[first_level], &centroid_infos, datas, query_dimension);
        PushQueue(centroid_infos, centroid_resource_s);
    } else { // 并行
        std::vector<std::vector<CentroidInfo>> con_infos;
        con_infos.resize(mercury_concurrency, std::vector<CentroidInfo>());

        size_t start = 0;
        std::vector<bthread_t> bthreads;
        std::vector<BthreadMessage> msgs;
        msgs.reserve(mercury_concurrency);
        for (int i = 0; i < mercury_concurrency; i++) {
            size_t end;
            if (i == mercury_concurrency - 1) {
                end = centroid_nums[first_level];
            } else {
                end = start + centroid_nums[first_level] / mercury_concurrency;
            }

            // 先改成直接启动线程
            bthread_t bid;
            BthreadMessage msg;
            msg.index = this;
            msg.centroid_resource = &centroid_resource;
            msg.start = start;
            msg.end = end;
            msg.centroid_infos = &con_infos.at(i);
            msg.datas = datas;
            msg.query_dimension = query_dimension;
            msgs.push_back(std::move(msg));
            if (bthread_start_background(&bid, NULL, BthreadRun, &msgs.at(msgs.size() - 1)) != 0) {
                LOG_ERROR("start bthread failed.");
                return -1;
            }
            bthreads.push_back(bid);

            start = end;
        }

        for (auto &t : bthreads) {
            bthread_join(t, NULL);
        }

        for (const auto &infos : con_infos) {
            PushQueue(infos, centroid_resource_s);
        }
    }

    group_real_slot_indexs.reserve(nprobe);
    size_t i = 0;
    while (!centroid_resource_s.empty() && i++ < nprobe) {
        auto index_label = centroid_resource_s.top().index;
        auto real_slot_index = centroid_resource_manager_.GetSlotIndex(group_index, index_label);
        group_real_slot_indexs.push_back(real_slot_index);
        centroid_resource_s.pop();
    }

    // 二级聚类
    if (EnableFineCluster()) {
        uint32_t fine_centroid_count = 0;
        // 统计中心点数量
        for (auto &coarse_slot_index : group_real_slot_indexs) {
            CentroidResource &fine_centroid_resource =
                fine_centroid_resource_manager_.GetCentroidResource(coarse_slot_index);
            fine_centroid_count += fine_centroid_resource.getLeafCentroidNum();
        }
        if (fine_centroid_count == 0) {
            LOG_ERROR("fine centroid_nums is empty");
            return -1;
        }
        std::vector<FineCentroidInfo> fine_centroid_infos;
        fine_centroid_infos.reserve(fine_centroid_count);
        // 计算中心点距离
        uint32_t start = 0;
        for (auto &coarse_slot_index : group_real_slot_indexs) {
            CentroidResource &fine_centroid_resource =
                fine_centroid_resource_manager_.GetCentroidResource(coarse_slot_index);
            CalcFineCentroidScore(fine_centroid_resource, 0, fine_centroid_resource.getLeafCentroidNum(),
                                  &fine_centroid_infos, datas, query_dimension, coarse_slot_index);
        }
        // 计算需要选出多少中心点，降级只在1级中心点里面做
        float fine_ratio = context_params.has(PARAM_FINE_SCAN_RATIO) ? context_params.getFloat(PARAM_FINE_SCAN_RATIO)
                                                                     : index_params.getFloat(PARAM_FINE_SCAN_RATIO);
        if (is_rt && is_multi_age && index_params.has(PARAM_RT_FINE_SCAN_RATIO)) {
            fine_ratio = context_params.has(PARAM_RT_FINE_SCAN_RATIO)
                             ? context_params.getFloat(PARAM_RT_FINE_SCAN_RATIO)
                             : index_params.getFloat(PARAM_RT_FINE_SCAN_RATIO);
        }
        size_t fine_nprobe = std::max<size_t>(static_cast<size_t>(fine_ratio * fine_centroid_count), 1);
        std::nth_element(fine_centroid_infos.begin(), fine_centroid_infos.begin() + fine_nprobe,
                         fine_centroid_infos.end());
        // 最终把选出来的二级中心点赋
        std::vector<off_t> fine_group_real_slot_indexs;
        fine_group_real_slot_indexs.reserve(fine_nprobe);
        for (size_t i = 0; i < fine_nprobe; i++) {
            auto real_slot_index = fine_centroid_resource_manager_.GetSlotIndex(fine_centroid_infos[i].coarse_index,
                                                                                fine_centroid_infos[i].index);
            fine_group_real_slot_indexs.push_back(real_slot_index);
        }
        group_real_slot_indexs = fine_group_real_slot_indexs;
    }

    std::sort(group_real_slot_indexs.begin(), group_real_slot_indexs.end());

    return 0;
}

void GroupIvfIndex::RecoverPostingFromSlot(std::vector<CoarseIndex<SmallBlock>::PostingIterator> &postings,
                                           std::vector<uint32_t> &ivf_postings_group_ids,
                                           std::vector<std::vector<off_t>> &real_slot_indexs,
                                           std::vector<uint32_t> &group_doc_nums, bool need_truncate,
                                           bool is_multi_query)
{
    const CoarseIndex<SmallBlock> &coarse_index = GetCoarseIndex();
    uint32_t truncate_num =
        index_params_.has(PARAM_GROUP_IVF_VISIT_LIMIT) ? index_params_.getUint32(PARAM_GROUP_IVF_VISIT_LIMIT) : 0;
    for (size_t i = 0; i < real_slot_indexs.size(); ++i) {
        postings.reserve(postings.size() + real_slot_indexs[i].size());
        uint32_t posting_num = 0;
        for (auto &group_real_slot_index : real_slot_indexs[i]) {
            CoarseIndex<SmallBlock>::PostingIterator posting = coarse_index.search(group_real_slot_index);
            posting_num += posting.getDocNum();
            if (need_truncate && truncate_num != 0 && posting_num >= truncate_num) {
                LOG_WARN("Realtime posting truncated, posting_num: %u, truncate_num: %u", posting_num, truncate_num);
                break;
            }
            group_doc_nums[i] += posting.getDocNum();
            postings.push_back(posting);
            ivf_postings_group_ids.push_back(is_multi_query ? i : 0);
        }
    }
}

int GroupIvfIndex::SearchGroup(docid_t docid, std::vector<GroupInfo> &group_infos, std::vector<int> &labels) const
{
    for (auto i = 0; i < coarse_index_.getHeader()->slotNum; ++i) {
        CoarseIndex<SmallBlock>::PostingIterator iter = coarse_index_.search(i);
        while (!iter.finish()) {
            if (docid == iter.next()) {
                labels.push_back(i);
                gindex_t group_index = centroid_resource_manager_.GetGroupIndex(i);
                group_infos.push_back(group_manager_.GetGroupInfo(group_index));
            }
        }
    }

    return 0;
}

MERCURY_NAMESPACE_END(core);
