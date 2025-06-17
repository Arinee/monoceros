#include "ivf_index.h"
#include "../hdfs_file_wrapper.h"
#include "src/core/utils/index_meta_helper.h"
#include "src/core/utils/string_util.h"
#include "../query_info.h"

MERCURY_NAMESPACE_BEGIN(core);

using namespace fslib;
using namespace fslib::fs;

MatrixPointer NullMatrix() {
    return MatrixPointer(nullptr);
}

int IvfIndex::Load(const void* data, size_t size) {
    if (Index::Load(data, size) != 0) {
        LOG_ERROR("Failed to call Index::Load.");
        return -1;
    }

    auto *component = index_package_.get(COMPONENT_ROUGH_MATRIX);
    if (!component) {
        LOG_ERROR("get component error: %s", COMPONENT_ROUGH_MATRIX);
        return -1;
    }
    if (!centroid_resource_.init((void *)component->getData(), component->getDataSize())) {
        LOG_ERROR("centroid resource init error");
        return -1;
    }

    component = index_package_.get(COMPONENT_COARSE_INDEX);
    if (!component) {
        LOG_ERROR("get component error: %s", COMPONENT_COARSE_INDEX);
        return -1;
    }
    if (!coarse_index_.load((void*)component->getData(), component->getDataSize())) {
        LOG_ERROR("coarse index load error");
        return -1;
    }

    //为了兼容老索引，后续下掉
    component = index_package_.get(COMPONENT_REDINDEX_DOCID_ARRAY);
    if (!component) {
        LOG_WARN("no REDINDEX_DOCID_ARRAY component.");
    } else {
        std::vector<int32_t> redindex_docids;
        auto ptr = static_cast<decltype(redindex_docids)::value_type*>(component->getData());
        redindex_docids.assign(ptr, ptr + component->getDataSize() / sizeof(decltype(redindex_docids)::value_type));
        coarse_index_.getHeader()->usedDocNum = redindex_docids.size();
    }

    component = index_package_.get(COMPONENT_SLOT_INDEX_PROFILE);
    if (!component) {
        LOG_ERROR("get component error: %s", COMPONENT_SLOT_INDEX_PROFILE);
        return -1;
    }
    if (!slot_index_profile_.load((void*)component->getData(), component->getDataSize())) {
        LOG_ERROR("slot index profle load error");
        return -1;
    }

    return 0;
}

int IvfIndex::CopyInit(const Index* index, size_t doc_num) {
    if (!index) {
        LOG_ERROR("Invalid index pointer.");
        return -1;
    }

    if (Index::CopyInit(index, doc_num) != 0) {
        LOG_ERROR("Failed to call Index::CopyInit.");
        return -1;
    }

    const IvfIndex* ivf_index = dynamic_cast<const IvfIndex*>(index);
    size_t max_doc_num = doc_num + GetReserverdDocNum();
    LOG_INFO("copy init max_doc_num: %ld", max_doc_num);

    centroid_resource_ = ivf_index->centroid_resource_;
    auto slot_num = centroid_resource_.getLeafCentroidNum();
    auto capacity = CoarseIndex<BigBlock>::calcSize(slot_num, max_doc_num);
    coarse_base_.assign(capacity, 0);
    if (!coarse_index_.create(coarse_base_.data(), capacity, slot_num, max_doc_num)) {
        LOG_ERROR("Failed to create CoarseIndex.");
        return -1;
    }

    size_t elem_size = sizeof(SlotIndex);
    capacity = ArrayProfile::CalcSize(max_doc_num, elem_size);
    slot_index_profile_base_.assign(capacity, 0);

    if (!slot_index_profile_.create(slot_index_profile_base_.data(),
                                    capacity, elem_size)) {
        LOG_ERROR("Failed to slot index profile");
        return -1;
    }

    return 0;
}

int IvfIndex::Create(IndexParams& index_params) {
    Index::SetIndexParams(index_params);
    if (!IndexMetaHelper::parseFrom(index_params.getString(PARAM_DATA_TYPE),
                                    index_params.getString(PARAM_METHOD),
                                    index_params.getUint64(PARAM_DIMENSION),
                                    index_meta_)) {
        LOG_ERROR("Failed to init ivf index meta.");
        return -1;
    }

    if (index_params.getString(PARAM_TRAIN_DATA_PATH) != "") {
        if (!InitIvfCentroidMatrix(index_params.getString(PARAM_TRAIN_DATA_PATH))) {
            LOG_ERROR("Faile to load ivf centroid matrix");
            return -1;
        }
    }

    auto elem_size = index_meta_.sizeofElement();
    auto mem_quota = index_params.getUint64(PARAM_GENERAL_INDEX_MEMORY_QUOTA);
    if (mem_quota == 0) {
        LOG_WARN("memory quota is not set, using default 1GB memory quota.");
        mem_quota = 1L * 1024L * 1024L * 1024L;
    }
    auto elem_count = MemQuota2DocCount(mem_quota, elem_size);
    LOG_INFO("---------- Containable element count: %lu", elem_count);
    if (Index::Create(elem_count) != 0) {
        LOG_ERROR("call index::create failed.");
        return -1;
    }

    //init CoarseIndex
    auto slotNum = centroid_resource_.getLeafCentroidNum();
    auto capacity = CoarseIndex<BigBlock>::calcSize(slotNum, max_doc_num_);
    coarse_base_.assign(capacity, 0);
    if (!coarse_index_.create(coarse_base_.data(), capacity, slotNum, max_doc_num_)) {
        LOG_ERROR("Failed to create CoarseIndex.");
        return -1;
    }

    elem_size = sizeof(SlotIndex);
    capacity = ArrayProfile::CalcSize(max_doc_num_, elem_size);
    slot_index_profile_base_.assign(capacity, 0);
    if (!slot_index_profile_.create(slot_index_profile_base_.data(),
                                    capacity, elem_size)) {
        LOG_ERROR("Failed to create slot index profile");
        return -1;
    }

    return 0;
}

MatrixPointer IvfIndex::DoLoadCentroidMatrix(const std::string& file_path, size_t dimension,
                                                    size_t element_size, size_t& centroid_size) const {
    std::string file_content;
    if (!HdfsFileWrapper::AtomicLoad(file_path, file_content)) {
        LOG_ERROR("load centroid file failed.");
        return NullMatrix();
    }

    std::vector<std::string> centroid_vec = StringUtil::split(file_content, "\n");
    centroid_size = centroid_vec.size();
    char* centroid_matrix = new char[element_size * centroid_vec.size()];
    if (centroid_matrix == nullptr) {
        LOG_ERROR("centroid_matrix is null");
        return NullMatrix();
    }

    MatrixPointer matrix_pointer(centroid_matrix);

    //loop every centroid
    for (size_t i = 0 ; i < centroid_vec.size(); i++) {
        std::vector<std::string> centroid_item = StringUtil::split(centroid_vec.at(i), " ");
        if (centroid_item.size() != 2) {
            LOG_ERROR("centroid_item space split format error: %s", centroid_vec.at(i).c_str());
            return NullMatrix();
        }

        std::vector<string> values = StringUtil::split(centroid_item.at(1), ",");
        if (values.size() != dimension) {
            LOG_ERROR("centroid value commar split format error: %s", centroid_item.at(1).c_str());
            return NullMatrix();
        }

        // support binary vector
        if (index_meta_.type() == IndexMeta::FeatureTypes::kTypeBinary) {
            std::vector<char> bit_values;
            bit_values.assign(element_size, 0);
            //loop every dimension
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
                    LOG_ERROR("invalid centroid vector value: %s, larger than 1.0, is not two-value", values.at(j).c_str());
                    return NullMatrix();
                }
            }
            memcpy((char*)centroid_matrix + element_size * i, bit_values.data(), element_size);
        } else {
            //loop every dimension
            for (size_t j = 0; j < values.size(); j++) {
                char value[32];//here at most int64_t 8 Byte, 32 is enough
                if(!StrToValue(values.at(j), (void*)value)) {
                    LOG_ERROR("centroid value str to value format error: %s", values.at(j).c_str());
                    return NullMatrix();
                }

                size_t type_size = element_size / dimension;
                memcpy((char*)centroid_matrix + element_size * i + j * type_size, (void*)value, type_size);
            }
        }
    }

    return std::move(matrix_pointer);
}

bool IvfIndex::StrToValue(const std::string& source, void* value) const {
    switch (index_meta_.type()) {
    case IndexMeta::FeatureTypes::kTypeUnknown:
        return false;
    case IndexMeta::FeatureTypes::kTypeBinary:
        //TODO, how to deal with binary
        return false;
    case IndexMeta::FeatureTypes::kTypeHalfFloat:
        if (!StringUtil::strToHalf(source.c_str(), *(half_float::half*)value)) {
            return false;
        }
        return true;
    case IndexMeta::FeatureTypes::kTypeFloat:
        if (!StringUtil::strToFloat(source.c_str(), *(float*)value)) {
            return false;
        }
        return true;
    case IndexMeta::FeatureTypes::kTypeDouble:
        if (!StringUtil::strToDouble(source.c_str(), *(double*)value)) {
            return false;
        }
        return true;
    case IndexMeta::FeatureTypes::kTypeInt8:
        if (!StringUtil::strToInt8(source.c_str(), *(int8_t*)value)) {
            return false;
        }
        return true;
    case IndexMeta::FeatureTypes::kTypeInt16:
        if (!StringUtil::strToInt16(source.c_str(), *(int16_t*)value)) {
            return false;
        }
        return true;
    }

    return false;
}

bool IvfIndex::InitIvfCentroidMatrix(const std::string& centroid_dir) {
    size_t centroid_size = 0;
    std::string centroid_path = centroid_dir + IVF_CENTROID_FILE_POSTFIX;
    MatrixPointer matrix_pointer = DoLoadCentroidMatrix(centroid_path, index_meta_.dimension(),
                                                        index_meta_.sizeofElement(), centroid_size);
    if (!matrix_pointer) {
        LOG_ERROR("Failed to Load centroid Matrix.");
        return false;
    }

    char* centroid_matrix = matrix_pointer.get();
    std::vector<uint32_t> centroids_levelcnts;
    centroids_levelcnts.push_back(centroid_size);

    //only support one level cnt, DefaultLevelCnt = 1.
    CentroidResource::RoughMeta rough_meta(index_meta_.sizeofElement(), DefaultLevelCnt, centroids_levelcnts);
    if (!centroid_resource_.create(rough_meta)) {
        LOG_ERROR("Failed to create centroid resource.");
        return false;
    }

    //only 1 level
    for (size_t i = 0; i < centroid_size; i++) {
        if (!centroid_resource_.setValueInRoughMatrix(0, i,
                                                      centroid_matrix + i * index_meta_.sizeofElement())) {
            LOG_ERROR("Failed to set centroid resource rough matrix.");
            return false;
        }
    }

    return true;
}

/// add a new vectoV
int IvfIndex::Add(docid_t doc_id, pk_t pk,
                  const std::string& query_str, 
                  const std::string& primary_key) {
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

    return Add(doc_id, pk, query_info.GetVector(), query_info.GetVectorLen());
}

int IvfIndex::Add(docid_t doc_id, pk_t pk, const void *val, size_t len) {
    if (Index::Add(doc_id, pk, val, len) != 0) {
        LOG_ERROR("failed to call Index::Add. doc_id: %u, pk: %lu, len: %lu", doc_id, pk, len);
        return -1;
    }

    if ((size_t)doc_id >= GetMaxDocNum()) {
        LOG_ERROR("docid should be less than GetMaxDocNum");
        return -1;
    }
    SlotIndex label = GetNearestLabel(val, len);
    if (label == INVALID_SLOT_INDEX) {
        LOG_ERROR("Failed to get nearest label.");
        return -1;
    }

    if (!coarse_index_.addDoc(label, doc_id)) {
        LOG_ERROR("Failed to add into coarse index.");
        return -1;
    }

    if (!slot_index_profile_.insert(doc_id, &label)) {
        LOG_ERROR("Failed to add into slot index profile.");
        return -1;
    }

    return 0;
}

int IvfIndex::AddBase(docid_t doc_id, pk_t pk) {
    return Index::Add(doc_id, pk, nullptr, 0);
}

SlotIndex IvfIndex::GetNearestLabel(const void* data, size_t /* size */) {
    if (!centroid_resource_.IsInit()) {
        LOG_ERROR("centroid resource is not init yet.");
        return INVALID_SLOT_INDEX;
    }
    SlotIndex return_labels = INVALID_SLOT_INDEX;
    size_t level = 0;
    auto roughMeta = centroid_resource_.getRoughMeta();
    std::vector<size_t> levelScanLimit;
    for (size_t i = 0; i < centroid_resource_.getRoughMeta().levelCnt - 1; ++i) {
        levelScanLimit.push_back(centroid_resource_.getRoughMeta().centroidNums[i] / 10);
    }

    std::priority_queue<CentroidInfo, std::vector<CentroidInfo>, std::greater<CentroidInfo>> centroids;

    for (uint32_t i = 0; roughMeta.centroidNums.size() && i < roughMeta.centroidNums[level]; ++i) {
        const void *centroidValue = centroid_resource_.getValueInRoughMatrix(level, i);
        float score = index_meta_.distance(data, centroidValue);
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
                const void *centroidValue = centroid_resource_.getValueInRoughMatrix(level, centroid);
                float dist = index_meta_.distance(data, centroidValue);
                centroids.emplace(centroid, dist);
            }
        }
    }

    if (!centroids.empty()) {
        return_labels = centroids.top().index;
    }
    return return_labels;
}

/// whither index segment full
bool IvfIndex::IsFull() const {
    //here current doc num从coarseIndex里面拿
    return GetDocNum() >= GetMaxDocNum();
}

float IvfIndex::GetRankScore(docid_t doc_id) {
    return static_cast<float>(*(SlotIndex*)slot_index_profile_.getInfo(doc_id));
}

size_t IvfIndex::MemQuota2DocCount(size_t memQuota, size_t /* elemSize */) const {
    int64_t slotNum = centroid_resource_.getLeafCentroidNum();
    size_t elemCount = 0;
    size_t realMemUsed = 0;
    do {
        elemCount += MIN_BUILD_COUNT;
        realMemUsed = 0;
        realMemUsed += CoarseIndex<BigBlock>::calcSize(slotNum, elemCount);
        realMemUsed += ArrayProfile::CalcSize(elemCount, sizeof(SlotIndex));
    } while (realMemUsed <= memQuota);
    elemCount -= MIN_BUILD_COUNT;
    if (elemCount == 0) {
        std::cerr << "memQuota: " << memQuota << ", elemCount: " << elemCount << std::endl;
    }

    return elemCount;
}

int IvfIndex::CompactIndex() {
    compact_index_ = std::make_shared<IvfIndex>();
    size_t total_doc_num = GetDocNum();
    compact_index_->CopyInit(this, total_doc_num);
    LOG_INFO("Total doc num in index: %lu", total_doc_num);

    for (docid_t docid = 0; docid < total_doc_num; docid++) {
        if (CompactEach(docid, compact_index_.get()) != 0) {
            return -1;
        }
    }

    return 0;
}

int IvfIndex::CompactEach(docid_t docid, IvfIndex* compact_index) {
    if (compact_index == nullptr) {
        LOG_ERROR("NULL compact index.");
        return -1;
    }

    SlotIndex slot_index = *(SlotIndex*)slot_index_profile_.getInfo(docid);

    bool ret = compact_index->GetCoarseIndex().addDoc(slot_index, docid);
    if (!ret) {
        LOG_ERROR("insert doc[%u] into coarse_index error", docid);
        return -1;
    }

    ret = compact_index->GetSlotIndexProfile().insert(docid, &slot_index);
    if (!ret) {
        LOG_ERROR("insert doc[%u] into slot_index_profile error ", docid);
        return -1;
    }

    return 0;
}

int IvfIndex::Dump(const void*& data, size_t& size) {
    if (CompactIndex() != 0) {
        LOG_ERROR("compact index failed.");
        return -1;
    }

    if (DumpHelper::DumpIvf(compact_index_.get(), index_package_) != 0) {
        return -1;
    }

    if (!index_package_.dump(data, size)) {
        LOG_ERROR("Failed to dump package.");
        return -1;
    }

    return 0;
}

int64_t IvfIndex::UsedMemoryInCurrent() const {
    return sizeof(*this) + coarse_index_.getHeader()->usedSize;
}

int IvfIndex::SearchIvf(std::vector<CoarseIndex<BigBlock>::PostingIterator>& postings, const void* data, size_t size, const IndexParams& context_params) {
    const IndexParams& index_params = GetIndexParams();
    CentroidResource& centroid_resource = GetCentroidResource();
    const IndexMeta& index_meta = GetIndexMeta();
    const CoarseIndex<BigBlock>& coarse_index = GetCoarseIndex();

    auto ratio = context_params.has(PARAM_COARSE_SCAN_RATIO) ? context_params.getFloat(PARAM_COARSE_SCAN_RATIO)
        : index_params.getFloat(PARAM_COARSE_SCAN_RATIO);
    // auto ratio = index_params.getFloat(PARAM_COARSE_SCAN_RATIO);
    LOG_INFO("scan ratio: %f", ratio);
    auto nprobe = static_cast<size_t>(ratio * centroid_resource.getLeafCentroidNum());
    if (nprobe < 1) {
        LOG_ERROR("nprobe is less than 1, set to 1.");
        nprobe = 1;
    }

    std::vector<uint32_t> coarse_index_labels;
    coarse_index_labels.reserve(nprobe);
    auto& rough_meta = centroid_resource.getRoughMeta();
    std::priority_queue<CentroidInfo, std::vector<CentroidInfo>, std::greater<CentroidInfo>> centroid_resource_s;
    auto &centroid_nums = rough_meta.centroidNums;
    if (centroid_nums.empty()) {
        LOG_ERROR("centroid_nums is empty");
        return -1;
    }

    const size_t first_level = 0;
    for (uint32_t i = 0; i < centroid_nums[first_level]; ++i)
    {
        const void* centroid_value = centroid_resource.getValueInRoughMatrix(first_level, i);
        float score = index_meta.distance(data, centroid_value);
        centroid_resource_s.emplace(i, score);
    }

    while(!centroid_resource_s.empty() && coarse_index_labels.size() < nprobe)
    {
        coarse_index_labels.push_back(centroid_resource_s.top().index);
        centroid_resource_s.pop();
    }

    std::sort(coarse_index_labels.begin(), coarse_index_labels.end());
    for (const auto& e : coarse_index_labels) {
        postings.push_back(coarse_index.search(e));
    }

    return 0;
}

MERCURY_NAMESPACE_END(core);
