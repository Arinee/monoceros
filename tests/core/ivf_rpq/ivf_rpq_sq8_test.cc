/// Copyright (c) 2020, xiaohongshu Inc. All rights reserved.
/// Author: sunan <sunan@xiaohongshu.com>
/// Created: 2020-08-27 12:00

#include <gtest/gtest.h>
#include <iostream>
#include <fstream>

#define protected public
#define private public
#include "src/core/algorithm/ivf_rpq/ivf_rpq_builder.h"
#include "src/core/algorithm/ivf_rpq/ivf_rpq_searcher.h"
#include "src/core/algorithm/ivf_rpq/ivf_rpq_merger.h"
#include "src/core/algorithm/algorithm_factory.h"
#include "src/core/algorithm/query_info.h"
#include "../group_ivf/common.h"
#include "src/core/framework/index_storage.h"
#include "src/core/framework/instance_factory.h"
#include "src/core/utils/string_util.h"
#undef protected
#undef private

MERCURY_NAMESPACE_BEGIN(core);

class IvfRPQSQ8Test: public testing::Test
{
public:
    void SetUp()
        {
            std::cout << "cwd is:" << getcwd(NULL, 0) << std::endl;
            index_params_.set(PARAM_COARSE_SCAN_RATIO, 0.5);
            index_params_.set(PARAM_METHOD, "L2");
            index_params_.set(PARAM_INDEX_TYPE, "IvfRpq");
            index_params_.set(PARAM_GENERAL_CONTAIN_FEATURE_PROFILE, true);
            index_params_.set(PARAM_DATA_TYPE, "float");
            index_params_.set(PARAM_DIMENSION, 64);
        }

    void TearDown()
        {
        }

    AlgorithmFactory factory_;
    IndexParams index_params_;
};

enum RangeStat { RS_minmax };

void train_Uniform(RangeStat rs, float rs_arg, size_t n, int k, const float* x, std::vector<float>& trained);

class Codec8bit {
public:
    static void encode_component(float x, uint8_t* code, int i);
    static float decode_component(const uint8_t* code, int i);
};

void Codec8bit::encode_component(float x, uint8_t* code, int i) {
    code[i] = static_cast<uint8_t>(255 * x);
}

float Codec8bit::decode_component(const uint8_t* code, int i) {
    return (code[i] + 0.5f) / 255.0f;
}
struct Quantizer {
    float vmin;
    float vmax;
    int d;

    void decode_vector(const uint8_t* code, float* x) const {
        float vdiff = vmax - vmin;
        for (int i = 0; i < d; ++i) {
            float xi = Codec8bit::decode_component(code, i);
            x[i] = vmin + xi * vdiff;
        }
    }

    void encode_vector(const float* x, uint8_t* code) const {
        float vdiff = vmax - vmin;
        for (int i = 0; i < d; ++i) {
            float xi = (x[i] - vmin) / vdiff;
            if (xi < 0) xi = 0;
            if (xi > 1.0) xi = 1.0;
            Codec8bit::encode_component(xi, code, i);
        }
    }
};

void train_Uniform(RangeStat rs, float rs_arg, size_t n, int k, const float* x, std::vector<float>& trained) {
    trained.resize(2);
    float& vmin = trained[0];
    float& vmax = trained[1];

    if (rs == RS_minmax) {
        vmin = HUGE_VAL;
        vmax = -HUGE_VAL;
        for (size_t i = 0; i < n; i++) {
            if (x[i] < vmin) vmin = x[i];
            if (x[i] > vmax) vmax = x[i];
        }
        float vexp = (vmax - vmin) * rs_arg;
        vmin -= vexp;
        vmax += vexp;
    }
}

void print_vector(const std::vector<float>& vec, const std::string& name) {
    std::cout << name << ":\n";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << vec[i] << " ";
        if ((i + 1) % 10 == 0) std::cout << "\n"; // 每10个元素换行一次
    }
    std::cout << "\n";
}

void print_uint8_vector(const std::vector<uint8_t>& vec, const std::string& name) {
    std::cout << name << ":\n";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << static_cast<int>(vec[i]) << " ";
        if ((i + 1) % 10 == 0) std::cout << "\n"; // 每10个元素换行一次
    }
    std::cout << "\n";
}

TEST_F(IvfRPQSQ8Test, TestBase) {
    const size_t num_vectors = 10; // 向量的数量
    const size_t d = 64; // 向量的维度
    std::vector<std::vector<float>> vectors(num_vectors, std::vector<float>(d));
    
    // 随机生成数据
    srand(time(NULL)); // 初始化随机数种子
    for (size_t i = 0; i < num_vectors; ++i) {
        for (size_t j = 0; j < d; ++j) {
            vectors[i][j] = static_cast<float>(rand()) / RAND_MAX * 6.0 - 3.0; // 范围[-3, 3]
        }
    }

    // 打印原始向量
    print_vector(vectors[0], "Original vector");

    // 将所有向量展平到一个数组中以便于训练
    std::vector<float> flat_vector(d * num_vectors);
    for (size_t i = 0; i < num_vectors; ++i) {
        memcpy(flat_vector.data() + i * d, vectors[i].data(), d * sizeof(float));
    }

    // 训练均匀量化器
    std::vector<float> trained;
    train_Uniform(RS_minmax, 0.0f, num_vectors * d, 256, flat_vector.data(), trained);
    float vmin = trained[0];
    float vmax = trained[1];

    // 初始化量化器
    Quantizer quantizer{vmin, vmax, static_cast<int>(d)};

    // 编码第一个向量
    std::vector<uint8_t> encoded_vector(d);
    quantizer.encode_vector(vectors[0].data(), encoded_vector.data());

    // 打印编码后的向量
    print_uint8_vector(encoded_vector, "Encoded vector");

    // 解码向量
    std::vector<float> decoded_vector(d);
    quantizer.decode_vector(encoded_vector.data(), decoded_vector.data());

    // 打印解码后的向量
    print_vector(decoded_vector, "Decoded vector");

    // 计算误差
    double mse = 0;
    for (size_t i = 0; i < d; ++i) {
        double diff = vectors[0][i] - decoded_vector[i];
        mse += diff * diff;
    }
    mse /= d;
    std::cout << "Mean Squared Error: " << mse << std::endl;
}

TEST_F(IvfRPQSQ8Test, TestFP32Lvl1) {
    index_params_.set(PARAM_TRAIN_DATA_PATH, "tests/core/ivf_rpq/test_data_residual");
    factory_.SetIndexParams(index_params_);
    Index::Pointer index = factory_.CreateIndex();
    mercury::core::IvfRpqIndex* core_index = dynamic_cast<mercury::core::IvfRpqIndex*>(index.get());
    std::string data_str1 = "0.0393844 0.0070153 0.0300562 0.0108158 0.1513422 0.0623903 -0.0106355 0.0964897 -0.0778688 0.0313834 -0.0071482 -0.1789710 -0.0245972 -0.0837628 -0.0700559 0.1637720 -0.0990320 -0.0146512 0.0582540 0.1287557 0.0083320 0.0438362 -0.0413511 -0.0553498 -0.0546915 0.1676616 0.0764068 -0.1951676 0.1394591 0.1202289 -0.0003936 -0.0332190 -0.0722720 0.0256493 0.0039293 0.0582260 -0.0959043 0.0888708 0.1429843 0.0581214 -0.0423285 0.0143103 -0.5067487 -0.0056066 -0.0577092 0.0168444 -0.1062441 0.0463585 0.0406294 0.1731417 0.1062065 0.0635848 -0.0001631 -0.0853588 -0.0219815 -0.0575943 -0.0092908 -0.1075933 0.0446303 0.0637256 0.1319047 0.1010149 0.1079177 0.0413118";
    std::string group_str1 = "0:0||" + data_str1;
    size_t size = core_index->index_meta_._element_size;
    int ret = core_index->Add(0, INVALID_PK, group_str1);
    std::string data_str2 = "-0.0928967 -0.0438082 0.0102797 0.0822286 -0.0055833 -0.0134031 0.0009328 0.0566829 -0.0122154 0.0462377 -0.1059280 -0.1328204 0.0872460 -0.0671041 -0.0705529 -0.0910953 -0.0291398 -0.0709173 0.0689292 0.0259000 -0.1163263 0.1032536 -0.0220323 0.0362972 -0.0154788 0.0032540 0.1409850 -0.0593337 -0.0146975 -0.0257620 0.1327830 -0.2349198 0.0213850 0.0301328 0.0632988 0.0307327 -0.0810744 0.0355883 0.0357120 -0.0477258 -0.0191721 0.0282200 -0.5539329 -0.0634755 -0.1209998 -0.0324256 0.1541371 0.0133156 -0.0669477 0.2175741 -0.0240443 -0.0440047 0.0034319 -0.1436657 -0.1078534 0.0220657 -0.0237252 -0.0384541 0.0737352 0.1846743 0.2189968 0.1087057 0.0757952 0.0485931";
    std::string group_str2 = "0:0||" + data_str2;
    ret = core_index->Add(1, INVALID_PK, group_str2);
    ASSERT_EQ(0, ret);
    ASSERT_EQ(2, core_index->GetDocNum());
    MergeIdMap id_map;
    id_map.push_back(std::make_pair(0, 0));
    id_map.push_back(std::make_pair(0, 1));

    std::vector<Index::Pointer> indexes;
    indexes.push_back(index);

    Merger::Pointer index_merger = factory_.CreateMerger();
    int suc = index_merger->MergeByDocid(indexes, id_map);
    ASSERT_EQ(0, suc);
    LOG_INFO("merge success");

    Searcher::Pointer index_searcher = factory_.CreateSearcher();
    size_t dump_size;
    const void* dump_data = index_merger->DumpIndex(&dump_size);
    ASSERT_TRUE(dump_data != nullptr);
    int ret_code = index_searcher->LoadIndex(dump_data, dump_size);
    ASSERT_EQ(0, ret_code);
    LOG_INFO("load index success");

    ASSERT_EQ(index_searcher->getFType(), IndexMeta::kTypeFloat);

    IndexParams index_params;
    mercury::core::GeneralSearchContext context(index_params);
    
    std::string search_str = "2&0:0#2||" + data_str2;
    QueryInfo query_info(search_str);
    query_info.SetFeatureTypes(IndexMeta::FeatureTypes::kTypeFloat);
    ASSERT_TRUE(query_info.MakeAsSearcher());
    ret_code = index_searcher->Search(query_info, &context);
    ASSERT_EQ(ret_code, 0);
    std::cout << "context.Result().at(0).gloid = " << context.Result().at(0).gloid << " and context.Result().at(0).score = " << context.Result().at(0).score << std::endl;
    std::cout << "context.Result().at(1).gloid = " << context.Result().at(1).gloid << " and context.Result().at(1).score = " << context.Result().at(1).score << std::endl;
    ASSERT_EQ(2, context.Result().size());
    LOG_INFO("ivf rpq search success");
}

TEST_F(IvfRPQSQ8Test, TestINT8Lvl1) {
    index_params_.set(PARAM_TRAIN_DATA_PATH, "tests/core/ivf_rpq/test_data_residual");
    index_params_.set(PARAM_ENABLE_QUANTIZE, true);
    factory_.SetIndexParams(index_params_);
    Index::Pointer index = factory_.CreateIndex();
    mercury::core::IvfRpqIndex* core_index = dynamic_cast<mercury::core::IvfRpqIndex*>(index.get());
    std::string data_str1 = "0.0393844 0.0070153 0.0300562 0.0108158 0.1513422 0.0623903 -0.0106355 0.0964897 -0.0778688 0.0313834 -0.0071482 -0.1789710 -0.0245972 -0.0837628 -0.0700559 0.1637720 -0.0990320 -0.0146512 0.0582540 0.1287557 0.0083320 0.0438362 -0.0413511 -0.0553498 -0.0546915 0.1676616 0.0764068 -0.1951676 0.1394591 0.1202289 -0.0003936 -0.0332190 -0.0722720 0.0256493 0.0039293 0.0582260 -0.0959043 0.0888708 0.1429843 0.0581214 -0.0423285 0.0143103 -0.5067487 -0.0056066 -0.0577092 0.0168444 -0.1062441 0.0463585 0.0406294 0.1731417 0.1062065 0.0635848 -0.0001631 -0.0853588 -0.0219815 -0.0575943 -0.0092908 -0.1075933 0.0446303 0.0637256 0.1319047 0.1010149 0.1079177 0.0413118";
    std::string group_str1 = "0:0||" + data_str1;
    size_t size = core_index->index_meta_._element_size;
    int ret = core_index->Add(0, INVALID_PK, group_str1);
    std::string data_str2 = "-0.0928967 -0.0438082 0.0102797 0.0822286 -0.0055833 -0.0134031 0.0009328 0.0566829 -0.0122154 0.0462377 -0.1059280 -0.1328204 0.0872460 -0.0671041 -0.0705529 -0.0910953 -0.0291398 -0.0709173 0.0689292 0.0259000 -0.1163263 0.1032536 -0.0220323 0.0362972 -0.0154788 0.0032540 0.1409850 -0.0593337 -0.0146975 -0.0257620 0.1327830 -0.2349198 0.0213850 0.0301328 0.0632988 0.0307327 -0.0810744 0.0355883 0.0357120 -0.0477258 -0.0191721 0.0282200 -0.5539329 -0.0634755 -0.1209998 -0.0324256 0.1541371 0.0133156 -0.0669477 0.2175741 -0.0240443 -0.0440047 0.0034319 -0.1436657 -0.1078534 0.0220657 -0.0237252 -0.0384541 0.0737352 0.1846743 0.2189968 0.1087057 0.0757952 0.0485931";
    std::string group_str2 = "0:0||" + data_str2;
    ret = core_index->Add(1, INVALID_PK, group_str2);
    ASSERT_EQ(0, ret);
    ASSERT_EQ(2, core_index->GetDocNum());
    MergeIdMap id_map;
    id_map.push_back(std::make_pair(0, 0));
    id_map.push_back(std::make_pair(0, 1));

    std::vector<Index::Pointer> indexes;
    indexes.push_back(index);

    Merger::Pointer index_merger = factory_.CreateMerger();
    int suc = index_merger->MergeByDocid(indexes, id_map);
    ASSERT_EQ(0, suc);
    LOG_INFO("merge success");

    Searcher::Pointer index_searcher = factory_.CreateSearcher();
    size_t dump_size;
    const void* dump_data = index_merger->DumpIndex(&dump_size);
    ASSERT_TRUE(dump_data != nullptr);
    int ret_code = index_searcher->LoadIndex(dump_data, dump_size);
    ASSERT_EQ(0, ret_code);
    LOG_INFO("load index success");

    ASSERT_EQ(index_searcher->getFType(), IndexMeta::kTypeFloat);

    IndexParams index_params;
    mercury::core::GeneralSearchContext context(index_params);
    
    std::string search_str = "2&0:0#2||" + data_str2;
    QueryInfo query_info(search_str);
    query_info.SetFeatureTypes(IndexMeta::FeatureTypes::kTypeFloat);
    ASSERT_TRUE(query_info.MakeAsSearcher());
    ret_code = index_searcher->Search(query_info, &context);
    ASSERT_EQ(ret_code, 0);
    std::cout << "context.Result().at(0).gloid = " << context.Result().at(0).gloid << " and context.Result().at(0).score = " << context.Result().at(0).score << std::endl;
    std::cout << "context.Result().at(1).gloid = " << context.Result().at(1).gloid << " and context.Result().at(1).score = " << context.Result().at(1).score << std::endl;
    ASSERT_EQ(2, context.Result().size());
    LOG_INFO("ivf rpq search success");
}

TEST_F(IvfRPQSQ8Test, TestFP32Lvl2) {
    index_params_.set(PARAM_ENABLE_FINE_CLUSTER, true);
    index_params_.set(PARAM_TRAIN_DATA_PATH, "tests/core/ivf_rpq/test_data_two_stage_residual");
    index_params_.set(PARAM_FINE_SCAN_RATIO, 0.5);
    factory_.SetIndexParams(index_params_);
    Index::Pointer index = factory_.CreateIndex();
    mercury::core::IvfRpqIndex* core_index = dynamic_cast<mercury::core::IvfRpqIndex*>(index.get());
    std::string data_str1 = "0.0926264 0.0079578 0.0105586 0.0418111 0.1812230 0.0563133 -0.0899557 0.1065935 -0.0799954 0.0870304 0.0170577 -0.2035020 -0.0675575 -0.0591432 -0.1244585 0.1755335 -0.1775552 -0.0377039 0.0819606 0.0970977 0.0279189 0.0369875 -0.0192580 0.0029287 -0.0357301 0.1562776 0.1798831 -0.1910289 0.1729529 0.1310860 0.0251708 -0.0850961 -0.0402595 0.0584627 0.0042594 0.0411005 -0.0828406 0.0533206 0.1465107 0.0839182 -0.0228154 0.0684170 -0.5612501 0.0055404 -0.0606350 0.0044111 -0.0993920 -0.0074073 0.1077544 0.1560783 0.1391315 0.1358985 -0.0099312 -0.0318797 -0.0148738 -0.0368931 -0.0088994 -0.1230934 0.0407985 0.0562747 0.1737005 0.0836068 0.0971618 0.0989029";
    std::string group_str1 = "0:0||" + data_str1;
    size_t size = core_index->index_meta_._element_size;
    int ret = core_index->Add(0, INVALID_PK, group_str1);
    std::string data_str2 = "0.0471769 0.0070062 -0.0272478 -0.0008411 0.1456381 0.1187976 -0.0170354 0.1569180 -0.0013319 0.0034049 -0.0252147 -0.1240342 -0.0177105 -0.0506851 -0.0470940 0.1735980 0.0264503 -0.1691135 0.0386053 0.1134081 0.0219201 0.0273431 -0.0604741 0.0064885 -0.0301704 0.1541573 0.1130584 -0.2313167 0.1558160 0.2252170 0.0446254 -0.0252525 -0.1035084 -0.0755953 -0.0021249 0.0895299 -0.1275391 0.0850627 0.2085826 0.1004199 -0.1149546 0.0522189 -0.5112755 0.0428439 -0.0510297 -0.0168479 -0.1283535 0.1035291 0.0465996 0.1657004 0.1109575 0.0038713 -0.0364446 -0.0948055 -0.0337404 -0.0509140 0.0225536 -0.1068570 0.0221448 0.1191636 0.0965643 0.1269277 0.0324667 0.0226748";
    std::string group_str2 = "0:0||" + data_str2;
    ret = core_index->Add(1, INVALID_PK, group_str2);
    std::string data_str3 = "-0.0569022 -0.0082786 -0.0013107 0.0322052 0.0190016 0.0418793 0.0045911 0.0666534 0.0198776 0.0693688 -0.1237503 -0.1748899 0.0788442 -0.0729922 0.0123018 -0.0257479 -0.0276087 -0.0679842 0.0226628 0.0465526 -0.0157705 0.1574665 -0.0255337 0.0217879 -0.0267191 0.0605810 0.1191377 -0.0885173 -0.0269630 -0.0219755 0.0802599 -0.1859989 0.0303423 -0.0001548 0.0569395 0.0569086 -0.0655188 0.0344526 0.0416267 -0.0519145 -0.0065157 0.0762323 -0.6201381 -0.0949297 -0.1091450 -0.0046707 0.1979933 0.0364175 -0.0781303 0.2057350 0.0252539 -0.0124358 -0.0596333 -0.1621402 -0.1037850 0.0198722 0.0488082 -0.0377144 0.0455353 0.1876951 0.1947537 0.0886972 0.0502329 0.0246371";
    std::string group_str3 = "0:0||" + data_str3;
    ret = core_index->Add(2, INVALID_PK, group_str3);
    ASSERT_EQ(0, ret);
    ASSERT_EQ(3, core_index->GetDocNum());
    MergeIdMap id_map;
    id_map.push_back(std::make_pair(0, 0));
    id_map.push_back(std::make_pair(0, 1));
    id_map.push_back(std::make_pair(0, 2));

    std::vector<Index::Pointer> indexes;
    indexes.push_back(index);

    Merger::Pointer index_merger = factory_.CreateMerger();
    int suc = index_merger->MergeByDocid(indexes, id_map);
    ASSERT_EQ(0, suc);
    LOG_INFO("merge success");

    Searcher::Pointer index_searcher = factory_.CreateSearcher();
    size_t dump_size;
    const void* dump_data = index_merger->DumpIndex(&dump_size);
    ASSERT_TRUE(dump_data != nullptr);
    int ret_code = index_searcher->LoadIndex(dump_data, dump_size);
    ASSERT_EQ(0, ret_code);
    LOG_INFO("load index success");

    ASSERT_EQ(index_searcher->getFType(), IndexMeta::kTypeFloat);

    IndexParams index_params;
    mercury::core::GeneralSearchContext context(index_params);
    
    std::string search_str = "3&0:0#3||" + data_str1;
    QueryInfo query_info(search_str);
    query_info.SetFeatureTypes(IndexMeta::FeatureTypes::kTypeFloat);
    ASSERT_TRUE(query_info.MakeAsSearcher());
    ret_code = index_searcher->Search(query_info, &context);
    ASSERT_EQ(ret_code, 0);
    ASSERT_EQ(3, context.Result().size());
    std::cout << "context.Result().at(0).gloid = " << context.Result().at(0).gloid << " and context.Result().at(0).score = " << context.Result().at(0).score << std::endl;
    std::cout << "context.Result().at(1).gloid = " << context.Result().at(1).gloid << " and context.Result().at(1).score = " << context.Result().at(1).score << std::endl;
    std::cout << "context.Result().at(2).gloid = " << context.Result().at(2).gloid << " and context.Result().at(2).score = " << context.Result().at(2).score << std::endl;
    LOG_INFO("ivf rpq search success");
}

TEST_F(IvfRPQSQ8Test, TestINT8Lvl2) {
    index_params_.set(PARAM_ENABLE_QUANTIZE, true);
    index_params_.set(PARAM_ENABLE_FINE_CLUSTER, true);
    index_params_.set(PARAM_TRAIN_DATA_PATH, "tests/core/ivf_rpq/test_data_two_stage_residual");
    index_params_.set(PARAM_FINE_SCAN_RATIO, 0.5);
    factory_.SetIndexParams(index_params_);
    Index::Pointer index = factory_.CreateIndex();
    mercury::core::IvfRpqIndex* core_index = dynamic_cast<mercury::core::IvfRpqIndex*>(index.get());
    std::string data_str1 = "0.0926264 0.0079578 0.0105586 0.0418111 0.1812230 0.0563133 -0.0899557 0.1065935 -0.0799954 0.0870304 0.0170577 -0.2035020 -0.0675575 -0.0591432 -0.1244585 0.1755335 -0.1775552 -0.0377039 0.0819606 0.0970977 0.0279189 0.0369875 -0.0192580 0.0029287 -0.0357301 0.1562776 0.1798831 -0.1910289 0.1729529 0.1310860 0.0251708 -0.0850961 -0.0402595 0.0584627 0.0042594 0.0411005 -0.0828406 0.0533206 0.1465107 0.0839182 -0.0228154 0.0684170 -0.5612501 0.0055404 -0.0606350 0.0044111 -0.0993920 -0.0074073 0.1077544 0.1560783 0.1391315 0.1358985 -0.0099312 -0.0318797 -0.0148738 -0.0368931 -0.0088994 -0.1230934 0.0407985 0.0562747 0.1737005 0.0836068 0.0971618 0.0989029";
    std::string group_str1 = "0:0||" + data_str1;
    size_t size = core_index->index_meta_._element_size;
    int ret = core_index->Add(0, INVALID_PK, group_str1);
    std::string data_str2 = "0.0471769 0.0070062 -0.0272478 -0.0008411 0.1456381 0.1187976 -0.0170354 0.1569180 -0.0013319 0.0034049 -0.0252147 -0.1240342 -0.0177105 -0.0506851 -0.0470940 0.1735980 0.0264503 -0.1691135 0.0386053 0.1134081 0.0219201 0.0273431 -0.0604741 0.0064885 -0.0301704 0.1541573 0.1130584 -0.2313167 0.1558160 0.2252170 0.0446254 -0.0252525 -0.1035084 -0.0755953 -0.0021249 0.0895299 -0.1275391 0.0850627 0.2085826 0.1004199 -0.1149546 0.0522189 -0.5112755 0.0428439 -0.0510297 -0.0168479 -0.1283535 0.1035291 0.0465996 0.1657004 0.1109575 0.0038713 -0.0364446 -0.0948055 -0.0337404 -0.0509140 0.0225536 -0.1068570 0.0221448 0.1191636 0.0965643 0.1269277 0.0324667 0.0226748";
    std::string group_str2 = "0:0||" + data_str2;
    ret = core_index->Add(1, INVALID_PK, group_str2);
    std::string data_str3 = "-0.0569022 -0.0082786 -0.0013107 0.0322052 0.0190016 0.0418793 0.0045911 0.0666534 0.0198776 0.0693688 -0.1237503 -0.1748899 0.0788442 -0.0729922 0.0123018 -0.0257479 -0.0276087 -0.0679842 0.0226628 0.0465526 -0.0157705 0.1574665 -0.0255337 0.0217879 -0.0267191 0.0605810 0.1191377 -0.0885173 -0.0269630 -0.0219755 0.0802599 -0.1859989 0.0303423 -0.0001548 0.0569395 0.0569086 -0.0655188 0.0344526 0.0416267 -0.0519145 -0.0065157 0.0762323 -0.6201381 -0.0949297 -0.1091450 -0.0046707 0.1979933 0.0364175 -0.0781303 0.2057350 0.0252539 -0.0124358 -0.0596333 -0.1621402 -0.1037850 0.0198722 0.0488082 -0.0377144 0.0455353 0.1876951 0.1947537 0.0886972 0.0502329 0.0246371";
    std::string group_str3 = "0:0||" + data_str3;
    ret = core_index->Add(2, INVALID_PK, group_str3);
    ASSERT_EQ(0, ret);
    ASSERT_EQ(3, core_index->GetDocNum());
    MergeIdMap id_map;
    id_map.push_back(std::make_pair(0, 0));
    id_map.push_back(std::make_pair(0, 1));
    id_map.push_back(std::make_pair(0, 2));

    std::vector<Index::Pointer> indexes;
    indexes.push_back(index);

    Merger::Pointer index_merger = factory_.CreateMerger();
    int suc = index_merger->MergeByDocid(indexes, id_map);
    ASSERT_EQ(0, suc);
    LOG_INFO("merge success");

    Searcher::Pointer index_searcher = factory_.CreateSearcher();
    size_t dump_size;
    const void* dump_data = index_merger->DumpIndex(&dump_size);
    ASSERT_TRUE(dump_data != nullptr);
    int ret_code = index_searcher->LoadIndex(dump_data, dump_size);
    ASSERT_EQ(0, ret_code);
    LOG_INFO("load index success");

    ASSERT_EQ(index_searcher->getFType(), IndexMeta::kTypeFloat);

    IndexParams index_params;
    mercury::core::GeneralSearchContext context(index_params);
    
    std::string search_str = "3&0:0#3||" + data_str1;
    QueryInfo query_info(search_str);
    query_info.SetFeatureTypes(IndexMeta::FeatureTypes::kTypeFloat);
    ASSERT_TRUE(query_info.MakeAsSearcher());

    ret_code = index_searcher->Search(query_info, &context);

    ASSERT_EQ(ret_code, 0);
    ASSERT_EQ(3, context.Result().size());
    std::cout << "context.Result().at(0).gloid = " << context.Result().at(0).gloid << " and context.Result().at(0).score = " << context.Result().at(0).score << std::endl;
    std::cout << "context.Result().at(1).gloid = " << context.Result().at(1).gloid << " and context.Result().at(1).score = " << context.Result().at(1).score << std::endl;
    std::cout << "context.Result().at(2).gloid = " << context.Result().at(2).gloid << " and context.Result().at(2).score = " << context.Result().at(2).score << std::endl;
    LOG_INFO("ivf rpq search success");
}

MERCURY_NAMESPACE_END(core);
