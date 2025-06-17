/// Copyright (c) 2019, xiaohongshu Inc. All rights reserved.
/// Author: kailuo <kailuo@xiaohongshu.com>
/// Created: 2019-12-17 00:59

#include <fstream>
#include <gtest/gtest.h>
#include <iostream>

#define protected public
#define private public
#include "src/core/algorithm/algorithm_factory.h"
#include "src/core/algorithm/group_ivf/group_ivf_builder.h"
#include "src/core/algorithm/query_info.h"
#include "src/core/framework/index_storage.h"
#include "src/core/framework/instance_factory.h"
#undef protected
#undef private

MERCURY_NAMESPACE_BEGIN(core);

class GroupIvfBuilderTest : public testing::Test
{
public:
    void SetUp()
    {
        index_params_.set(PARAM_COARSE_SCAN_RATIO, 0.5);
        index_params_.set(PARAM_TRAIN_DATA_PATH, "tests/core/group_ivf/test_data/");
        index_params_.set(PARAM_DATA_TYPE, "float");
        index_params_.set(PARAM_METHOD, "L2");
        index_params_.set(PARAM_DIMENSION, 256);
        index_params_.set(PARAM_INDEX_TYPE, "GroupIvf");
        index_params_.set(PARAM_GENERAL_CONTAIN_FEATURE_PROFILE, true);
        factory_.SetIndexParams(index_params_);
    }

    void TearDown() {}

    AlgorithmFactory factory_;
    IndexParams index_params_;
};

TEST_F(GroupIvfBuilderTest, TestCalcScore)
{
    IndexParams index_params;
    index_params.set(PARAM_COARSE_SCAN_RATIO, 0.5);
    index_params.set(PARAM_TRAIN_DATA_PATH, "tests/core/group_ivf/test_data/");
    index_params.set(PARAM_DATA_TYPE, "float");
    index_params.set(PARAM_METHOD, "L2");
    index_params.set(PARAM_DIMENSION, 256);
    index_params.set(PARAM_INDEX_TYPE, "GroupIvf");
    index_params.set(PARAM_GENERAL_MAX_BUILD_NUM, 10000);
    index_params.set(PARAM_GENERAL_CONTAIN_FEATURE_PROFILE, true);
    mercury::core::GroupIvfBuilder builder;
    builder.Init(index_params);
    ASSERT_FLOAT_EQ(0, builder.CalcScore(0, 0));
    ASSERT_FLOAT_EQ(2.3025851, builder.CalcScore(0, 9));
    ASSERT_FLOAT_EQ(9.0109129, builder.CalcScore(0, 8191));
}

TEST_F(GroupIvfBuilderTest, TestCalcScoreBySpecifiedGroupLevel)
{
    IndexParams index_params;
    index_params.set(PARAM_COARSE_SCAN_RATIO, 0.5);
    index_params.set(PARAM_TRAIN_DATA_PATH, "tests/core/group_ivf/test_data/");
    index_params.set(PARAM_DATA_TYPE, "float");
    index_params.set(PARAM_METHOD, "L2");
    index_params.set(PARAM_DIMENSION, 256);
    index_params.set(PARAM_INDEX_TYPE, "GroupIvf");
    index_params.set(PARAM_GENERAL_MAX_BUILD_NUM, 10000);
    index_params.set(PARAM_GENERAL_CONTAIN_FEATURE_PROFILE, true);
    index_params.set(PARAM_SORT_BUILD_GROUP_LEVEL, 1);
    mercury::core::GroupIvfBuilder builder;
    builder.Init(index_params);
    ASSERT_FLOAT_EQ(0, builder.CalcScore(0, 0));
    ASSERT_FLOAT_EQ(0, builder.CalcScore(0, 9));
    ASSERT_FLOAT_EQ(0, builder.CalcScore(0, 8191));
    ASSERT_FLOAT_EQ(2.3025851, builder.CalcScore(1, 9));
    ASSERT_FLOAT_EQ(9.0109129, builder.CalcScore(1, 8191));
}
/*
TEST_F(GroupIvfBuilderTest, TestLoadLocal) {
    AlgorithmFactory factory;
    IndexParams index_params;
    index_params.set(PARAM_COARSE_SCAN_RATIO, 1);
    index_params.set(PARAM_TRAIN_DATA_PATH, "tests/core/group_ivf/test_data/");
    index_params.set(PARAM_DATA_TYPE, "float");
    index_params.set(PARAM_METHOD, "IP");
    index_params.set(PARAM_DIMENSION, 128);
    index_params.set(PARAM_INDEX_TYPE, "GroupIvf");
    index_params.set(PARAM_GENERAL_CONTAIN_FEATURE_PROFILE, true);
    factory.SetIndexParams(index_params);

    IndexStorage::Pointer index_storage = InstanceFactory::CreateStorage("MMapFileStorage");
    ASSERT_TRUE(index_storage);
    IndexStorage::Handler::Pointer handler = index_storage->open("/data1/kailuo/access/ms/mercury_index.package",
false); ASSERT_TRUE(handler); void *data = nullptr; ASSERT_EQ(handler->read((const void **)(&data), handler->size()),
handler->size());

    std::cout<<"handler size: "<<handler->size()<<std::endl;

    //Searcher::Pointer searcher = factory.CreateBuilder();
    //searcher->LoadIndex(data, handler->size());
    //GroupIvfIndex& index = ((GroupIvfSearcher*)searcher.get())->index_;
    GroupIvfBuilder builder;
    builder.index_.Load(data, handler->size());

    std::ifstream in("/data1/kailuo/group_index/data_main_search/doc_file");
    std::ofstream out("/data1/kailuo/group_index/data_main_search/doc_file_score1");
    char buffer[5000];
    if (!in.is_open())
    { std::cout << "Error opening file"; exit (1); }
    while (!in.eof())
    {
        in.getline(buffer, 5000);
        //std::cout<<buffer<<std::endl;
        float score = 0.0;
        std::string buf_str = buffer;
        if (buf_str == "") {
            continue;
        }
        //std::cout<<buf_str<<std::endl;
        int ret = builder.GetRankScore(buf_str, &score);
        ASSERT_EQ(ret, 0);
        out << buffer <<"=" << score << std::endl;
    }

    in.close();
    out.close();
    ASSERT_EQ(1, 0);
    }*/
TEST_F(GroupIvfBuilderTest, TestQueryInfo)
{
    {
        std::string data_str =
            "-0.102875 -0.047437 0.016824 -0.032322 -0.018978 0.00603 0.004624 -0.042992 0.043804 -0.129003 -0.063305 "
            "-0.018899 0.047719 0.03726 -0.012384 0.024918 0.063164 0.003078 -0.054183 -0.118877 -0.054619 0.11011 "
            "0.096558 0.018475 -0.087097 0.026629 -0.08566 0.02428 -0.088073 -0.046569 -0.094956 -0.027088 0.034621 "
            "-0.106306 -0.008979 0.031298 0.030987 0.123223 -0.004162 -0.04761 -0.037318 0.039261 0.136555 0.019844 "
            "-0.042915 0.100461 -0.061183 0.028209 -0.084324 0.012819 0.028181 -0.08822 0.035264 -0.072143 -0.015044 "
            "0.111965 -0.031478 0.068231 -0.018672 0.001338 0.046578 0.067762 -0.057287 -0.047343 -0.005129 0.161173 "
            "-0.021491 -0.003186 0.015636 0.006977 0.003917 -0.031233 0.024885 0.071695 -0.05812 -0.099307 -0.061015 "
            "0.031348 0.074765 -0.005988 -0.036811 0.062878 -0.117832 -0.013208 0.037202 0.017278 0.132848 -0.028692 "
            "0.107519 0.04048 -0.07074 -0.038141 -0.116665 -0.049325 0.138709 0.02446 -0.006483 0.097473 0.019068 "
            "-0.030756 0.003464 -0.004049 -0.017734 -0.049131 0.011849 -0.087313 0.018132 0.093301 0.02402 -0.114743 "
            "-0.073973 0.022781 -0.003511 0.043055 0.069776 -0.050155 -0.018691 0.078315 -0.016702 0.042535 -0.088182 "
            "-0.055301 0.038858 0.051842 0.048028 -0.044823 -0.051279 -0.016467 0.046392 -0.060022 -0.057112 0.130293 "
            "0.130172 0.118126 -0.038441 0.104364 0.046031 0.078625 0.021274 -0.007293 -0.002615 -0.127438 -0.014907 "
            "-0.071469 0.010509 -0.061879 -0.074339 0.004105 0.010812 0.017881 0.03734 -0.043334 -0.099021 -0.007516 "
            "-0.050524 -0.006559 0.035337 0.075333 -0.072907 -0.06585 0.015818 -0.013102 -0.043893 -0.046585 0.030032 "
            "-0.080634 0.048787 0.053605 -0.025551 -0.014829 -0.113559 -0.01594 -0.081099 0.11146 0.001189 -0.051934 "
            "-0.014657 -0.06457 0.045306 -0.013705 0.02578 0.035293 -0.045513 0.017706 -0.013743 -0.036449 -0.097975 "
            "0.00361 0.110656 0.005985 0.003918 -0.083217 0.089543 0.064883 0.063359 -0.067709 0.149407 -0.019442 "
            "-0.001442 -0.09393 0.117456 0.079656 -0.028365 -0.054244 0.112843 -0.033993 -0.112373 7.26E-4 -0.05703 "
            "-0.04225 0.012609 -0.002963 0.076794 -0.004844 0.072359 -0.110624 0.057226 -0.045802 0.05446 0.03787 "
            "0.011126 -0.121707 -0.034402 -0.040453 0.008523 0.012866 -0.061767 -0.146913 -0.032847 0.020639 0.055404 "
            "-0.01983 -0.036355 -0.022834 -0.132642 0.007916 -0.089387 -0.003184 -0.019981 -0.097821 -0.026173 "
            "0.101657 0.099773 -0.084372 -0.0099 0.020376 0.012375 -0.082308 0.034197 0.003457 0.007728 -0.031867 "
            "-0.035034 -0.003641 0.021217 -0.076501";
        std::string group_str = "1:11;2:200||" + data_str;
        QueryInfo query_info(group_str); // groupinfos 对应的 topks
        query_info.MakeAsBuilder();
        ASSERT_EQ("1:11;2:200", query_info.GetRawGroupInfo());
    }

    {
        std::string data_str =
            "-0.133974 -0.0435357 0.0166469 0.0918716 -0.0723077 -0.126416 -0.130181 -1.41379 0.0134664 -0.0257569 "
            "-0.0101074 0.0725689 0.0281584 0.142158 -0.0127134 0.135168 0.110247 -0.149395 1.45207 -1.01204 1.40394 "
            "-0.107016 1.36572 -0.144818 -0.026861 -1.5313 0.0895916 -0.150838 -0.00486981 -0.0316218 0.0142393 "
            "0.0767939 0.101098 -0.177069 -0.186746 0.0951623 0.0167225 -0.0341906 -0.00552497 0.0356712 0.116786 "
            "-1.50554 0.0541926 -0.0173092 -0.0743573 0.0535408 0.0488294 -0.0853423 -0.0354825 -0.0282569 0.0773245 "
            "1.40658 -0.074884 0.0967988 0.0338275 -0.0842706 0.11348 -0.0570158 0.13139 -0.0345481 0.147282 -0.115792 "
            "0.00945218 -0.00985493 -0.481099 -0.0536982 0.0479875 -0.0944779 0.0324009 0.156092 -0.0311313 -0.0626294 "
            "-0.0351052 0.0918406 0.0324137 0.0712964 -0.0232633 -0.0473209 -0.0292101 -0.0514297 0.03534 0.0298944 "
            "0.057301 -0.0229562 0.141807 0.0180136 -0.13703 -0.0271773 0.0967978 -0.0676003 0.0851782 0.0990961 "
            "0.0590544 0.018955 -0.00405138 0.0476703 -0.0170976 -0.0169255 -0.142662 -0.142594 0.458787 -0.137927 "
            "-0.053962 0.0395505 0.0263989 -0.0374196 -0.0187221 0.137572 -0.481086 -0.011021 -0.139939 -0.126421 "
            "-0.0305266 0.132105 -0.443343 0.0512023 0.0512399 0.0292831 0.0523659 -0.000497819 -0.0301933 0.14379 "
            "0.0538012 -0.479285 0.00249586 0.125138 -0.0397789 -0.00323576 0.0141277 0.0749739 -0.0409434 0.0467658 "
            "0.0173196 0.0483841 -0.0405902 -0.0664604 -0.0729921 0.0147514 0.0201786 0.0467818 -0.0848263 0.0329447 "
            "0.106823 0.072499 -0.00306189 0.0348559 0.0751251 0.0161329 0.0160419 0.0186001 -0.0520563 -0.0185936 "
            "0.0222952 -0.0335103 0.0643959 -0.0052561 -0.116388 0.079519 0.0499593 -0.00228709 -0.0792127 0.00035618 "
            "-0.0886017 0.0514815 -0.0105844 0.0243577 0.0367084 -0.0549138 -0.0131758 0.0731761 -0.0221393 0.0226625 "
            "0.0144555 -0.0224273 -0.0440066 0.0291137 -0.0961877 -0.0210577 -0.0274991 0.00818293 -0.00531995 "
            "-0.0340459 0.0588481 -0.0644117 0.0130346 -0.0134578 -0.0729885 0.0170534 0.0370878 0.0932697 -0.0483207 "
            "9.58382e-05";
        std::string group_str = "1000&3:9654#10||" + data_str;
        QueryInfo query_info(group_str); // groupinfos 对应的 topks
        query_info.MakeAsSearcher();
        ASSERT_EQ("1000&3:9654#10", query_info.GetRawGroupInfo());
    }
}

TEST_F(GroupIvfBuilderTest, TestNormalUse)
{
    AlgorithmFactory factory;
    IndexParams index_params;
    index_params.set(PARAM_COARSE_SCAN_RATIO, 0.5);
    index_params.set(PARAM_TRAIN_DATA_PATH, "tests/core/group_ivf/test_data/");
    index_params.set(PARAM_DATA_TYPE, "float");
    index_params.set(PARAM_METHOD, "L2");
    index_params.set(PARAM_DIMENSION, 256);
    index_params.set(PARAM_INDEX_TYPE, "GroupIvf");
    index_params.set(PARAM_GENERAL_MAX_BUILD_NUM, 10000);
    index_params.set(PARAM_GENERAL_CONTAIN_FEATURE_PROFILE, true);
    factory.SetIndexParams(index_params);

    char *buffer;
    // 也可以将buffer作为输出参数
    buffer = getcwd(NULL, 0);
    std::cout << "cwd is:" << buffer << std::endl;

    Builder::Pointer builder_p = factory.CreateBuilder();
    mercury::core::GroupIvfBuilder *builder = dynamic_cast<mercury::core::GroupIvfBuilder *>(builder_p.get());

    ASSERT_TRUE(builder != nullptr);
    ASSERT_EQ(10000, builder->index_->GetMaxDocNum());

    std::string data_str =
        "-0.102875 -0.047437 0.016824 -0.032322 -0.018978 0.00603 0.004624 -0.042992 0.043804 -0.129003 -0.063305 "
        "-0.018899 0.047719 0.03726 -0.012384 0.024918 0.063164 0.003078 -0.054183 -0.118877 -0.054619 0.11011 "
        "0.096558 0.018475 -0.087097 0.026629 -0.08566 0.02428 -0.088073 -0.046569 -0.094956 -0.027088 0.034621 "
        "-0.106306 -0.008979 0.031298 0.030987 0.123223 -0.004162 -0.04761 -0.037318 0.039261 0.136555 0.019844 "
        "-0.042915 0.100461 -0.061183 0.028209 -0.084324 0.012819 0.028181 -0.08822 0.035264 -0.072143 -0.015044 "
        "0.111965 -0.031478 0.068231 -0.018672 0.001338 0.046578 0.067762 -0.057287 -0.047343 -0.005129 0.161173 "
        "-0.021491 -0.003186 0.015636 0.006977 0.003917 -0.031233 0.024885 0.071695 -0.05812 -0.099307 -0.061015 "
        "0.031348 0.074765 -0.005988 -0.036811 0.062878 -0.117832 -0.013208 0.037202 0.017278 0.132848 -0.028692 "
        "0.107519 0.04048 -0.07074 -0.038141 -0.116665 -0.049325 0.138709 0.02446 -0.006483 0.097473 0.019068 "
        "-0.030756 0.003464 -0.004049 -0.017734 -0.049131 0.011849 -0.087313 0.018132 0.093301 0.02402 -0.114743 "
        "-0.073973 0.022781 -0.003511 0.043055 0.069776 -0.050155 -0.018691 0.078315 -0.016702 0.042535 -0.088182 "
        "-0.055301 0.038858 0.051842 0.048028 -0.044823 -0.051279 -0.016467 0.046392 -0.060022 -0.057112 0.130293 "
        "0.130172 0.118126 -0.038441 0.104364 0.046031 0.078625 0.021274 -0.007293 -0.002615 -0.127438 -0.014907 "
        "-0.071469 0.010509 -0.061879 -0.074339 0.004105 0.010812 0.017881 0.03734 -0.043334 -0.099021 -0.007516 "
        "-0.050524 -0.006559 0.035337 0.075333 -0.072907 -0.06585 0.015818 -0.013102 -0.043893 -0.046585 0.030032 "
        "-0.080634 0.048787 0.053605 -0.025551 -0.014829 -0.113559 -0.01594 -0.081099 0.11146 0.001189 -0.051934 "
        "-0.014657 -0.06457 0.045306 -0.013705 0.02578 0.035293 -0.045513 0.017706 -0.013743 -0.036449 -0.097975 "
        "0.00361 0.110656 0.005985 0.003918 -0.083217 0.089543 0.064883 0.063359 -0.067709 0.149407 -0.019442 "
        "-0.001442 -0.09393 0.117456 0.079656 -0.028365 -0.054244 0.112843 -0.033993 -0.112373 7.26E-4 -0.05703 "
        "-0.04225 0.012609 -0.002963 0.076794 -0.004844 0.072359 -0.110624 0.057226 -0.045802 0.05446 0.03787 0.011126 "
        "-0.121707 -0.034402 -0.040453 0.008523 0.012866 -0.061767 -0.146913 -0.032847 0.020639 0.055404 -0.01983 "
        "-0.036355 -0.022834 -0.132642 0.007916 -0.089387 -0.003184 -0.019981 -0.097821 -0.026173 0.101657 0.099773 "
        "-0.084372 -0.0099 0.020376 0.012375 -0.082308 0.034197 0.003457 0.007728 -0.031867 -0.035034 -0.003641 "
        "0.021217 -0.076501";
    std::string group_str = "0:0;1:11;2:200||" + data_str;
    QueryInfo query_info(group_str); // groupinfos 对应的 topks
    query_info.MakeAsBuilder();

    gindex_t group_index = 0;
    docid_t docid = 0;
    int ret = builder->AddDoc(docid, INVALID_PK, group_str);
    ASSERT_EQ(0, ret);

    const void *dump_content = nullptr;
    size_t dump_size = 0;

    dump_content = builder->DumpIndex(&dump_size);
    ASSERT_TRUE(dump_content != nullptr);
    ASSERT_TRUE(dump_size != 0);

    Index::Pointer loaded_index_p = factory_.CreateIndex(true);
    loaded_index_p->Load(dump_content, dump_size);
    GroupIvfIndex *core_index = dynamic_cast<GroupIvfIndex *>(loaded_index_p.get());

    SlotIndex label = 2;
    EXPECT_EQ(label, core_index->GetNearestGroupLabel(query_info.GetVector(), query_info.GetVectorLen(), group_index,
                                                      core_index->GetCentroidResourceManager()));
    core_index->coarse_index_.PrintStats();
    EXPECT_EQ(0, core_index->coarse_index_.search(label).next());
    EXPECT_EQ(1, core_index->GetDocNum());
    EXPECT_EQ(10000, core_index->GetMaxDocNum());
}

TEST_F(GroupIvfBuilderTest, TestMultAgeMode)
{
    AlgorithmFactory factory;
    IndexParams index_params;
    index_params.set(PARAM_COARSE_SCAN_RATIO, 0.5);
    index_params.set(PARAM_TRAIN_DATA_PATH, "tests/core/group_ivf/test_data/");
    index_params.set(PARAM_DATA_TYPE, "float");
    index_params.set(PARAM_METHOD, "L2");
    index_params.set(PARAM_DIMENSION, 256);
    index_params.set(PARAM_INDEX_TYPE, "GroupIvf");
    index_params.set(PARAM_GENERAL_MAX_BUILD_NUM, 10000);
    index_params.set(PARAM_GENERAL_CONTAIN_FEATURE_PROFILE, true);
    index_params.set(PARAM_MULTI_AGE_MODE, true);
    factory.SetIndexParams(index_params);

    char *buffer;
    // 也可以将buffer作为输出参数
    buffer = getcwd(NULL, 0);
    std::cout << "cwd is:" << buffer << std::endl;

    Builder::Pointer builder_p = factory.CreateBuilder();
    mercury::core::GroupIvfBuilder *builder = dynamic_cast<mercury::core::GroupIvfBuilder *>(builder_p.get());

    ASSERT_TRUE(builder != nullptr);
    ASSERT_EQ(10000, builder->index_->GetMaxDocNum());

    std::string data_str =
        "-0.102875 -0.047437 0.016824 -0.032322 -0.018978 0.00603 0.004624 -0.042992 0.043804 -0.129003 -0.063305 "
        "-0.018899 0.047719 0.03726 -0.012384 0.024918 0.063164 0.003078 -0.054183 -0.118877 -0.054619 0.11011 "
        "0.096558 0.018475 -0.087097 0.026629 -0.08566 0.02428 -0.088073 -0.046569 -0.094956 -0.027088 0.034621 "
        "-0.106306 -0.008979 0.031298 0.030987 0.123223 -0.004162 -0.04761 -0.037318 0.039261 0.136555 0.019844 "
        "-0.042915 0.100461 -0.061183 0.028209 -0.084324 0.012819 0.028181 -0.08822 0.035264 -0.072143 -0.015044 "
        "0.111965 -0.031478 0.068231 -0.018672 0.001338 0.046578 0.067762 -0.057287 -0.047343 -0.005129 0.161173 "
        "-0.021491 -0.003186 0.015636 0.006977 0.003917 -0.031233 0.024885 0.071695 -0.05812 -0.099307 -0.061015 "
        "0.031348 0.074765 -0.005988 -0.036811 0.062878 -0.117832 -0.013208 0.037202 0.017278 0.132848 -0.028692 "
        "0.107519 0.04048 -0.07074 -0.038141 -0.116665 -0.049325 0.138709 0.02446 -0.006483 0.097473 0.019068 "
        "-0.030756 0.003464 -0.004049 -0.017734 -0.049131 0.011849 -0.087313 0.018132 0.093301 0.02402 -0.114743 "
        "-0.073973 0.022781 -0.003511 0.043055 0.069776 -0.050155 -0.018691 0.078315 -0.016702 0.042535 -0.088182 "
        "-0.055301 0.038858 0.051842 0.048028 -0.044823 -0.051279 -0.016467 0.046392 -0.060022 -0.057112 0.130293 "
        "0.130172 0.118126 -0.038441 0.104364 0.046031 0.078625 0.021274 -0.007293 -0.002615 -0.127438 -0.014907 "
        "-0.071469 0.010509 -0.061879 -0.074339 0.004105 0.010812 0.017881 0.03734 -0.043334 -0.099021 -0.007516 "
        "-0.050524 -0.006559 0.035337 0.075333 -0.072907 -0.06585 0.015818 -0.013102 -0.043893 -0.046585 0.030032 "
        "-0.080634 0.048787 0.053605 -0.025551 -0.014829 -0.113559 -0.01594 -0.081099 0.11146 0.001189 -0.051934 "
        "-0.014657 -0.06457 0.045306 -0.013705 0.02578 0.035293 -0.045513 0.017706 -0.013743 -0.036449 -0.097975 "
        "0.00361 0.110656 0.005985 0.003918 -0.083217 0.089543 0.064883 0.063359 -0.067709 0.149407 -0.019442 "
        "-0.001442 -0.09393 0.117456 0.079656 -0.028365 -0.054244 0.112843 -0.033993 -0.112373 7.26E-4 -0.05703 "
        "-0.04225 0.012609 -0.002963 0.076794 -0.004844 0.072359 -0.110624 0.057226 -0.045802 0.05446 0.03787 0.011126 "
        "-0.121707 -0.034402 -0.040453 0.008523 0.012866 -0.061767 -0.146913 -0.032847 0.020639 0.055404 -0.01983 "
        "-0.036355 -0.022834 -0.132642 0.007916 -0.089387 -0.003184 -0.019981 -0.097821 -0.026173 0.101657 0.099773 "
        "-0.084372 -0.0099 0.020376 0.012375 -0.082308 0.034197 0.003457 0.007728 -0.031867 -0.035034 -0.003641 "
        "0.021217 -0.076501";
    std::string group_str = "0:0;1:11;2:200||" + data_str;
    QueryInfo query_info(group_str); // groupinfos 对应的 topks
    query_info.MakeAsBuilder();

    gindex_t group_index = 0;
    docid_t docid = 0;
    int ret = builder->AddDoc(docid++, INVALID_PK, group_str);
    ASSERT_EQ(0, ret);
    uint32_t create_time_s = *(uint32_t *)builder->index_->GetDocCreateTimeProfile().getInfo(0);
    ASSERT_EQ(0, create_time_s);

    ret = builder->AddDoc(docid++, INVALID_PK, group_str, "6489c4bf0000000027028f48");
    ASSERT_EQ(0, ret);
    create_time_s = *(uint32_t *)builder->index_->GetDocCreateTimeProfile().getInfo(1);
    ASSERT_EQ(1686750399, create_time_s);
}

TEST_F(GroupIvfBuilderTest, TestNoFeature)
{
    char *buffer;
    // 也可以将buffer作为输出参数
    buffer = getcwd(NULL, 0);
    std::cout << "cwd is:" << buffer << std::endl;

    Builder::Pointer builder_p = factory_.CreateBuilder();
    mercury::core::GroupIvfBuilder *builder = dynamic_cast<mercury::core::GroupIvfBuilder *>(builder_p.get());

    ASSERT_TRUE(builder != nullptr);

    std::string data_str =
        "-0.102875 -0.047437 0.016824 -0.032322 -0.018978 0.00603 0.004624 -0.042992 0.043804 -0.129003 -0.063305 "
        "-0.018899 0.047719 0.03726 -0.012384 0.024918 0.063164 0.003078 -0.054183 -0.118877 -0.054619 0.11011 "
        "0.096558 0.018475 -0.087097 0.026629 -0.08566 0.02428 -0.088073 -0.046569 -0.094956 -0.027088 0.034621 "
        "-0.106306 -0.008979 0.031298 0.030987 0.123223 -0.004162 -0.04761 -0.037318 0.039261 0.136555 0.019844 "
        "-0.042915 0.100461 -0.061183 0.028209 -0.084324 0.012819 0.028181 -0.08822 0.035264 -0.072143 -0.015044 "
        "0.111965 -0.031478 0.068231 -0.018672 0.001338 0.046578 0.067762 -0.057287 -0.047343 -0.005129 0.161173 "
        "-0.021491 -0.003186 0.015636 0.006977 0.003917 -0.031233 0.024885 0.071695 -0.05812 -0.099307 -0.061015 "
        "0.031348 0.074765 -0.005988 -0.036811 0.062878 -0.117832 -0.013208 0.037202 0.017278 0.132848 -0.028692 "
        "0.107519 0.04048 -0.07074 -0.038141 -0.116665 -0.049325 0.138709 0.02446 -0.006483 0.097473 0.019068 "
        "-0.030756 0.003464 -0.004049 -0.017734 -0.049131 0.011849 -0.087313 0.018132 0.093301 0.02402 -0.114743 "
        "-0.073973 0.022781 -0.003511 0.043055 0.069776 -0.050155 -0.018691 0.078315 -0.016702 0.042535 -0.088182 "
        "-0.055301 0.038858 0.051842 0.048028 -0.044823 -0.051279 -0.016467 0.046392 -0.060022 -0.057112 0.130293 "
        "0.130172 0.118126 -0.038441 0.104364 0.046031 0.078625 0.021274 -0.007293 -0.002615 -0.127438 -0.014907 "
        "-0.071469 0.010509 -0.061879 -0.074339 0.004105 0.010812 0.017881 0.03734 -0.043334 -0.099021 -0.007516 "
        "-0.050524 -0.006559 0.035337 0.075333 -0.072907 -0.06585 0.015818 -0.013102 -0.043893 -0.046585 0.030032 "
        "-0.080634 0.048787 0.053605 -0.025551 -0.014829 -0.113559 -0.01594 -0.081099 0.11146 0.001189 -0.051934 "
        "-0.014657 -0.06457 0.045306 -0.013705 0.02578 0.035293 -0.045513 0.017706 -0.013743 -0.036449 -0.097975 "
        "0.00361 0.110656 0.005985 0.003918 -0.083217 0.089543 0.064883 0.063359 -0.067709 0.149407 -0.019442 "
        "-0.001442 -0.09393 0.117456 0.079656 -0.028365 -0.054244 0.112843 -0.033993 -0.112373 7.26E-4 -0.05703 "
        "-0.04225 0.012609 -0.002963 0.076794 -0.004844 0.072359 -0.110624 0.057226 -0.045802 0.05446 0.03787 0.011126 "
        "-0.121707 -0.034402 -0.040453 0.008523 0.012866 -0.061767 -0.146913 -0.032847 0.020639 0.055404 -0.01983 "
        "-0.036355 -0.022834 -0.132642 0.007916 -0.089387 -0.003184 -0.019981 -0.097821 -0.026173 0.101657 0.099773 "
        "-0.084372 -0.0099 0.020376 0.012375 -0.082308 0.034197 0.003457 0.007728 -0.031867 -0.035034 -0.003641 "
        "0.021217 -0.076501";
    std::string group_str = "0:0;1:11;2:200||" + data_str;
    QueryInfo query_info(group_str); // groupinfos 对应的 topks
    query_info.MakeAsBuilder();

    gindex_t group_index = 0;
    docid_t docid = 0;
    int ret = builder->AddDoc(docid, INVALID_PK, group_str);
    ASSERT_EQ(0, ret);

    const void *dump_content = nullptr;
    size_t dump_size = 0;

    dump_content = builder->DumpIndex(&dump_size);
    ASSERT_TRUE(dump_content != nullptr);
    ASSERT_TRUE(dump_size != 0);

    Index::Pointer loaded_index_p = factory_.CreateIndex(true);
    loaded_index_p->Load(dump_content, dump_size);
    GroupIvfIndex *core_index = dynamic_cast<GroupIvfIndex *>(loaded_index_p.get());

    SlotIndex label = 2;
    EXPECT_EQ(label, core_index->GetNearestGroupLabel(query_info.GetVector(), query_info.GetVectorLen(), group_index,
                                                      core_index->GetCentroidResourceManager()));
    EXPECT_EQ(0, core_index->coarse_index_.search(label).next());
    EXPECT_EQ(1, core_index->GetDocNum());
    EXPECT_EQ(1040000, core_index->GetMaxDocNum());
}
MERCURY_NAMESPACE_END(core);
