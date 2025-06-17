/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     query_info.h
 *   \author   kailuo@xiaohongshu.com
 *   \date     2019.12.13
 *   \version  1.0.0
 *   \brief    build_str:   类目层级:c1;类目层级:c2||v1 v2 v3...
 *             search_str:  total&类目层级:c1#topk,c2#topk;类目层级:c3#topk||v1 v2 v3...;v1 v2 v3...||k1=v1,k2=v2
 *             v1 v2 v3... 兼容这种形式
 */

#pragma once
#include "group_manager.h"
#include "src/core/utils/string_util.h"
#include "src/core/framework/index_meta.h"

MERCURY_NAMESPACE_BEGIN(core);

struct AgeInfo {
    uint32_t age;
    uint32_t topk;
    AgeInfo() {};
    AgeInfo(uint32_t a, uint32_t k)
        : age(a), topk(k) {};
    bool operator<(const AgeInfo& right) const
    {
        return (age < right.age);
    }
};

class QueryInfo {
public:
    QueryInfo(const std::string& raw_str);

    QueryInfo();

    void SetQuery(const std::string& raw_str);

    void SetFeatureTypes(IndexMeta::FeatureTypes type);

    bool MakeAsBuilder();

    bool MakeAsSearcher();

    const void* GetVector() const;
    const void* GetBiasVector() const;

    const std::vector<const void*>& GetVectors() const;
    const std::vector<const void*>& GetBiasVectors() const;

    void SetTotalRecall(uint32_t total);
    void BoostTopkByRatio(float ratio);
    uint32_t GetTotalRecall() const;

    size_t GetVectorLen() const;
    size_t GetBiasVectorLen() const;

    size_t GetDimension() const;
    size_t GetBiasDimension() const;

    const std::vector<GroupInfo>& GetGroupInfos() const;

    void AddGroupInfo(uint32_t level, uint32_t id);

    const std::vector<uint32_t>& GetTopks() const;
    const std::vector<AgeInfo>& GetAgeInfos() const;

    const std::string& GetRawQuery() const;

    const std::string& GetRawGroupInfo() const;

    const IndexParams& GetContextParams() const;

    const bool ContainsZeroGroup() const;
    const bool MultiAgeMode() const;
    const bool MultiQueryMode() const;

    bool SetDowngradePercent(const std::string& downgradePercentStr);

private:
    bool MakeVector(const std::string& vec_str, std::vector<const void *> &data_pointers, size_t &dimension, size_t &element_size, std::unique_ptr<char[]> &vecs_pointer);

    bool ResolveCommon(std::vector<std::string>& level2_vec);

    //类目层级:c1
    bool ResolveBuildGroup(const std::string& target_str);

    //ageInSecond#topk
    bool ResolveAgeGroup(const std::string& target_str);

    //类目层级:c1#topk,c2#topk
    bool ResolveSearchGroup(const std::string& target_str);

    bool ResolveIndexParams(const std::string& params_str);

    bool AdaptIndexParam(const std::string& key, const std::string& value);

private:
    std::unique_ptr<char[]> vecs_pointer_;
    std::unique_ptr<char[]> bias_vecs_pointer_;
    std::vector<const void *> data_pointers_;
    std::vector<const void *> bias_data_pointers_;
    std::vector<GroupInfo> group_infos_;
    std::vector<AgeInfo> age_infos_;
    std::vector<uint32_t> topks_;
    uint32_t total_ = 0;
    std::string raw_str_;
    std::string raw_group_info_;
    IndexParams index_params_;

    bool contains_zero_group_ = false;
    bool multi_age_mode_ = false;
    bool multi_query_mode_ = false;

    IndexMeta::FeatureTypes type_;
    size_t dimension_;
    size_t element_size_;
    size_t bias_dimension_;
    size_t bias_element_size_;
};

MERCURY_NAMESPACE_END(core);
