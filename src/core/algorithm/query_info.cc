#include "query_info.h"
#include "src/core/utils/string_util.h"
#include "src/core/common/common_define.h"

MERCURY_NAMESPACE_BEGIN(core);

QueryInfo::QueryInfo(const std::string& raw_str)
    : raw_str_(raw_str) {
    type_ = IndexMeta::FeatureTypes::kTypeFloat;
}

QueryInfo::QueryInfo() {
}

const bool QueryInfo::ContainsZeroGroup() const {
    return contains_zero_group_;
}

const bool QueryInfo::MultiAgeMode() const {
    return multi_age_mode_;
}

const bool QueryInfo::MultiQueryMode() const {
    return multi_query_mode_;
}

void QueryInfo::SetQuery(const std::string& raw_str) {
    raw_str_ = raw_str;
}

void QueryInfo::SetFeatureTypes(IndexMeta::FeatureTypes type) {
    type_ = type;
}

const IndexParams& QueryInfo::GetContextParams() const {
    return index_params_;
}

const void* QueryInfo::GetVector() const {
    return data_pointers_.at(0);
}

const void* QueryInfo::GetBiasVector() const {
    return bias_data_pointers_.at(0);
}

const std::vector<const void*>& QueryInfo::GetVectors() const {
    return data_pointers_;
}

const std::vector<const void*>& QueryInfo::GetBiasVectors() const {
    return bias_data_pointers_;
}

void QueryInfo::SetTotalRecall(uint32_t total) {
    total_ = total;
}

void QueryInfo::BoostTopkByRatio(float ratio) {
    total_ *= ratio;
    if (multi_age_mode_) {
        for (auto& age_info : age_infos_) {
            age_info.topk *= ratio;
        }
        return;
    }
    for (auto& topk : topks_) {
        topk *= ratio;
    }
}

uint32_t QueryInfo::GetTotalRecall() const {
    return total_;
}

const std::string& QueryInfo::GetRawGroupInfo() const {
    return raw_group_info_;
}

size_t QueryInfo::GetVectorLen() const {
    return element_size_;
}

size_t QueryInfo::GetBiasVectorLen() const {
    return bias_element_size_;
}

size_t QueryInfo::GetDimension() const {
    return dimension_;
}

size_t QueryInfo::GetBiasDimension() const {
    return bias_dimension_;
}

const std::vector<GroupInfo>& QueryInfo::GetGroupInfos() const {
    return group_infos_;
}

void QueryInfo::AddGroupInfo(uint32_t level, uint32_t id) {
    group_infos_.push_back(GroupInfo(level, id));
}

const std::vector<uint32_t>& QueryInfo::GetTopks() const {
    return topks_;
}

const std::vector<AgeInfo>& QueryInfo::GetAgeInfos() const {
    return age_infos_;
}

const std::string& QueryInfo::GetRawQuery() const {
    return raw_str_;
}

bool QueryInfo::MakeAsBuilder() {
    std::vector<std::string> level2_vec;
    if (!ResolveCommon(level2_vec)) {
        LOG_ERROR("resolve common failed.");
        return false;
    }
    for (size_t i = 0; i < level2_vec.size(); i++) {
        if (!ResolveBuildGroup(level2_vec.at(i))) {
            LOG_ERROR("resolve build group failed.");
            return false;
        }
    }

    if (group_infos_.empty()) {
        group_infos_.emplace_back(0, 0);
    }

    return true;
}

bool QueryInfo::MakeAsSearcher() {
    std::vector<std::string> level2_vec;
    if (!ResolveCommon(level2_vec)) {
        LOG_ERROR("resolve common failed.");
        return false;
    }
    if (multi_age_mode_) {
        group_infos_.emplace_back(0, 0);
        return true;
    }
    for (size_t i = 0; i < level2_vec.size(); i++) {
        if (!ResolveSearchGroup(level2_vec.at(i))) {
            LOG_ERROR("resolve build group failed.");
            return false;
        }
    }

    return true;
}

bool QueryInfo::MakeVector(const std::string& vec_str, std::vector<const void *> &data_pointers, size_t &dimension, size_t &element_size, std::unique_ptr<char[]> &vecs_pointer) {
    std::vector<std::string> vecs = StringUtil::split(vec_str, ";");
    char* vecs_matrix;
    for (size_t j = 0; j < vecs.size(); j++) {
        std::vector<std::string> vec = StringUtil::split(vecs.at(j), " ");
        if (j == 0) {
            dimension = vec.size();
            element_size = IndexMeta::Sizeof(type_, dimension);
            vecs_matrix = new char[element_size * vecs.size()];
            if (vecs_matrix == nullptr) {
                LOG_ERROR("centroid_matrix is null");
                return false;
            }
            vecs_pointer.reset(vecs_matrix);
        }
        
        if (type_ == IndexMeta::FeatureTypes::kTypeFloat) {
            std::vector<float> data;
            for (size_t i = 0; i < vec.size(); i++) {
                float value;
                if (!StringUtil::strToFloat(vec.at(i).c_str(), value)) {
                    LOG_ERROR("convert to float failed. ori_str: %s", vec.at(i).c_str());
                    return false;
                }
                data.push_back(std::move(value));
            }
            memcpy(vecs_matrix + j * element_size, data.data(), element_size);
        } else if (type_ == IndexMeta::FeatureTypes::kTypeHalfFloat) {
            std::vector<half_float::half> data;
            for (size_t i = 0; i < vec.size(); i++) {
                half_float::half value;
                if (!StringUtil::strToHalf(vec.at(i).c_str(), value)) {
                    LOG_ERROR("convert to half float failed. ori_str: %s", vec.at(i).c_str());
                    return false;
                }
                data.push_back(std::move(value));
            }
            memcpy(vecs_matrix + j * element_size, data.data(), element_size);
        } else if (type_ == IndexMeta::FeatureTypes::kTypeBinary) {
            if (dimension % 8 != 0) {
                LOG_ERROR("FeatureType is Binary, but dimension mod 8 != 0");
                return false;
            }
            std::vector<char> bit_values;
            bit_values.assign(element_size, 0);
            //loop every dimension
            for (size_t i = 0; i < vec.size(); i++) {
                uint32_t pos = i % 8;
                if (vec.at(i) == "0") {
                    continue;
                } else if (vec.at(i) == "1") {
                    char mask = 0x1 << (7u - pos);
                    char &value = bit_values.at(i / 8);
                    value |= mask;
                } else {
                    LOG_ERROR("invalid query vector value: %s, is not two-value", vec.at(j).c_str());
                    return false;
                }
            }
            memcpy(vecs_matrix + j * element_size, bit_values.data(), element_size);
        } else {
            LOG_ERROR("Not support feature types to make vector.")
        }
        data_pointers.push_back(vecs_matrix + j * element_size);
    }

    return true;
}

bool QueryInfo::ResolveCommon(std::vector<std::string>& level2_vec) {
    std::vector<std::string> level1_vec = StringUtil::split(raw_str_, "||");

    if (level1_vec.size() < 1 || level1_vec.size() > 3) {
        LOG_ERROR("level1_vec resolve error. raw_str: %s", raw_str_.c_str());
        return false;
    }

    if (level1_vec.size() == 1) { //兼容无类目格式
        contains_zero_group_ = true;
        return MakeVector(level1_vec.at(0), data_pointers_, dimension_, element_size_, vecs_pointer_);
    }

    if (!MakeVector(level1_vec.at(1), data_pointers_, dimension_, element_size_, vecs_pointer_)) {
        return false;
    }
    if (level1_vec.size() >=3) {
        ResolveIndexParams(level1_vec.at(2));
    }

    if (multi_age_mode_) {
        return true;
    }

    raw_group_info_ = level1_vec.at(0);
    std::vector<std::string> total_vec = StringUtil::split(level1_vec.at(0), "&");
    std::vector<std::string> results;
    if (total_vec.size() != 1 && total_vec.size() != 2) {
        LOG_ERROR("total_vec resolve error. raw_str: %s", raw_str_.c_str());
        return false;
    }

    if (total_vec.size() == 1) {
        results = StringUtil::split(total_vec.at(0), ";");
    } else {
        results = StringUtil::split(total_vec.at(1), ";");
        if (!StringUtil::strToUInt32(total_vec.at(0).c_str(), total_)) {
            return false;
        }
    }

    level2_vec.assign(results.begin(), results.end());
    return true;
}

bool QueryInfo::ResolveIndexParams(const std::string& params_str) {
    //params_str: k1=v1,k2=v2
    std::vector<std::string> kvlist = StringUtil::split(params_str, ",");
    for (auto kv : kvlist) {
        std::vector<std::string> kvpair = StringUtil::split(kv, "=");
        if (kvpair.size() != 2) {
            LOG_ERROR("resolve kv_pair failed. params: %s", params_str.c_str());
            return false;
        }

        AdaptIndexParam(kvpair.at(0), kvpair.at(1));
    }

    return true;
}

bool QueryInfo::SetDowngradePercent(const std::string& downgradePercentStr) {
    float float_value = 0.0;
    if (!StringUtil::strToFloat(downgradePercentStr.c_str(), float_value)) {
        LOG_ERROR("convert downgradePercentStr %s to float failed.", downgradePercentStr.c_str());
        return false;
    }
    if (float_value > 2) {
        LOG_ERROR("downgrade percent[%f] large than 2, set to 2.0", float_value);
        float_value = 2.0;
    } else if (float_value < 0) {
        LOG_ERROR("downgrade percent[%f] less than 0, set to 0.0", float_value);
        float_value = 0.0;
    }
    return index_params_.set(PARAM_DOWNGRADE_PERCENT, float_value);
}

bool QueryInfo::AdaptIndexParam(const std::string& key, const std::string& value) {
    if (key == PARAM_COARSE_SCAN_RATIO) {
        float float_value = 0.0;
        if (!StringUtil::strToFloat(value.c_str(), float_value)) {
            LOG_ERROR("convert %s to float failed.", value.c_str());
            return false;
        }
        index_params_.set(PARAM_COARSE_SCAN_RATIO, float_value);
    } else if (key == PARAM_RT_COARSE_SCAN_RATIO) {
        float float_value = 0.0;
        if (!StringUtil::strToFloat(value.c_str(), float_value)) {
            LOG_ERROR("convert %s to float failed.", value.c_str());
            return false;
        }
        index_params_.set(PARAM_RT_COARSE_SCAN_RATIO, float_value);
    } else if (key == PARAM_FINE_SCAN_RATIO) {
        float float_value = 0.0;
        if (!StringUtil::strToFloat(value.c_str(), float_value)) {
            LOG_ERROR("convert %s to float failed.", value.c_str());
            return false;
        }
        index_params_.set(PARAM_FINE_SCAN_RATIO, float_value);
    } else if (key == PARAM_RT_FINE_SCAN_RATIO) {
        float float_value = 0.0;
        if (!StringUtil::strToFloat(value.c_str(), float_value)) {
            LOG_ERROR("convert %s to float failed.", value.c_str());
            return false;
        }
        index_params_.set(PARAM_RT_FINE_SCAN_RATIO, float_value);
    } else if (key == PARAM_GRAPH_MAX_SCAN_NUM_IN_QUERY) {
        int max_scan_num_in_query = 0;
        if (!StringUtil::strToInt32(value.c_str(), max_scan_num_in_query)) {
            LOG_ERROR("convert %s to int failed.", value.c_str());
            return false;
        }
        index_params_.set(PARAM_GRAPH_MAX_SCAN_NUM_IN_QUERY, max_scan_num_in_query);        
    } else if (key == PARAM_VAMANA_INDEX_SEARCH_L) {
        int vamana_index_search_l = 0;
        if (!StringUtil::strToInt32(value.c_str(), vamana_index_search_l)) {
            LOG_ERROR("convert %s to int failed.", value.c_str());
            return false;
        }
        index_params_.set(PARAM_VAMANA_INDEX_SEARCH_L, vamana_index_search_l);        
    } else if (key == PARAM_VAMANA_INDEX_SEARCH_BW) {
        int vamana_index_search_bw = 0;
        if (!StringUtil::strToInt32(value.c_str(), vamana_index_search_bw)) {
            LOG_ERROR("convert %s to int failed.", value.c_str());
            return false;
        }
        index_params_.set(PARAM_VAMANA_INDEX_SEARCH_BW, vamana_index_search_bw);        
    } else if (key == PARAM_BIAS_VECTOR) {
        if (!MakeVector(value, bias_data_pointers_, bias_dimension_, bias_element_size_, bias_vecs_pointer_)) {
            return false;
        }
    } else if (key == PARAM_MULTI_AGE_MODE) {
        index_params_.set(key, value);
        if (!ResolveAgeGroup(value)) {
            return false;
        }
    } else if (key == PARAM_MULTI_QUERY_MODE) {
        multi_query_mode_ = true;
        index_params_.set(key, value);
    } else {
        index_params_.set(key, value);
    }

    return true;
}

bool QueryInfo::ResolveAgeGroup(const std::string& target_str) {
    uint32_t age = 0, topk = 0;
    std::vector<std::string> age_confs = StringUtil::split(target_str, ";");
    if (age_confs.empty()) {
        return false;
    }
    age_infos_.reserve(age_confs.size());
    for (size_t i = 0; i < age_confs.size(); i++) {
        std::vector<std::string> age_conf = StringUtil::split(age_confs[i], "#");
        if (age_conf.size() != 2) {
            return false;
        }
        if (!StringUtil::strToUInt32(age_conf.at(0).c_str(), age)) {
            return false;
        }
        if (!StringUtil::strToUInt32(age_conf.at(1).c_str(), topk)) {
            return false;
        }
        if (age == 3600) { // 扩展1h到1.5h
            age = 5400;
        }
        age_infos_.emplace_back(age, topk);
        total_ += topk;
    }
    std::sort(age_infos_.begin(), age_infos_.end());
    multi_age_mode_ = true;
    return true;
}

//类目层级:c1
bool QueryInfo::ResolveBuildGroup(const std::string& target_str) {
    std::vector<std::string> level3_vec = StringUtil::split(target_str, ":");
    if (level3_vec.size() != 2) {
        return false;
    }

    GroupInfo info;
    if (!StringUtil::strToUInt32(level3_vec.at(0).c_str(), info.level)) {
        return false;
    }

    if (!StringUtil::strToUInt32(level3_vec.at(1).c_str(), info.id)) {
        return false;
    }

    group_infos_.push_back(info);
    return true;
}

//类目层级:c1#topk,c2#topk
bool QueryInfo::ResolveSearchGroup(const std::string& target_str) {
    std::vector<std::string> level3_vec = StringUtil::split(target_str, ":");
    if (level3_vec.size() != 2) {
        return false;
    }

    uint32_t level;
    if (!StringUtil::strToUInt32(level3_vec.at(0).c_str(), level)) {
        return false;
    }

    std::vector<std::string> level4_vec = StringUtil::split(level3_vec.at(1).c_str(), ",");
    for (size_t i = 0; i < level4_vec.size(); i++) {
        std::vector<std::string> level5_vec = StringUtil::split(level4_vec.at(i).c_str(), "#");
        if (level5_vec.size() != 2) {
            return false;
        }

        GroupInfo info;
        info.level = level;
        contains_zero_group_ |= (level==0);
        if (!StringUtil::strToUInt32(level5_vec.at(0).c_str(), info.id)) {
            return false;
        }
        group_infos_.push_back(info);

        uint32_t topk;
        if (!StringUtil::strToUInt32(level5_vec.at(1).c_str(), topk)) {
            return false;
        }
        topks_.push_back(topk);
    }

    return true;
}

MERCURY_NAMESPACE_END(core);
