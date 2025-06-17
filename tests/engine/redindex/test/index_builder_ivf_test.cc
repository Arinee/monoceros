/// Copyright (c) 2019, xiaohongshu Inc. All rights reserved.
/// Author: qiuming <qiuming@xiaohongshu.com>
/// Created: 2019-09-06 00:59

#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
//#include <sys/types.h>
//#include <sys/stat.h>
//#include <unistd.h>

#define protected public
#define private public
#include "src/engine/redindex/index_builder.h"
#include "src/engine/redindex/index_builder_factory.h"
#include "src/engine/redindex/redindex_factory.h"
#include "src/core/algorithm/ivf/ivf_index.h"
#include "src/core/algorithm/query_info.h"
#undef protected
#undef private

namespace mercury {
namespace redindex {
using namespace mercury::core;

class IndexBuilderIvfTest: public testing::Test
{
public:
    void SetUp()
    {
    }

    void TearDown()
    {
    }

};

TEST_F(IndexBuilderIvfTest, TestNormalUse)
{
    char *buffer;
    //也可以将buffer作为输出参数
    buffer = getcwd(NULL, 0);
    std::cout << "cwd is:" << buffer << std::endl;

    SchemaParams schema = {
        {"DataType", "float"},
        {"Method","L2"}, {"Dimension", "256"}, {"IndexType", "IvfFlat"},
        {"TrainDataPath", "tests/engine/redindex/test_data/coarse_8191"},
        {"IvfCoarseScanRatio", "0.5"}
    };

    IndexBuilderFactory factory;
    IndexBuilder::Pointer builder = factory.Create(schema);
    ASSERT_TRUE(!!builder == true);

    std::string data_str = "-0.102875 -0.047437 0.016824 -0.032322 -0.018978 0.00603 0.004624 -0.042992 0.043804 -0.129003 -0.063305 -0.018899 0.047719 0.03726 -0.012384 0.024918 0.063164 0.003078 -0.054183 -0.118877 -0.054619 0.11011 0.096558 0.018475 -0.087097 0.026629 -0.08566 0.02428 -0.088073 -0.046569 -0.094956 -0.027088 0.034621 -0.106306 -0.008979 0.031298 0.030987 0.123223 -0.004162 -0.04761 -0.037318 0.039261 0.136555 0.019844 -0.042915 0.100461 -0.061183 0.028209 -0.084324 0.012819 0.028181 -0.08822 0.035264 -0.072143 -0.015044 0.111965 -0.031478 0.068231 -0.018672 0.001338 0.046578 0.067762 -0.057287 -0.047343 -0.005129 0.161173 -0.021491 -0.003186 0.015636 0.006977 0.003917 -0.031233 0.024885 0.071695 -0.05812 -0.099307 -0.061015 0.031348 0.074765 -0.005988 -0.036811 0.062878 -0.117832 -0.013208 0.037202 0.017278 0.132848 -0.028692 0.107519 0.04048 -0.07074 -0.038141 -0.116665 -0.049325 0.138709 0.02446 -0.006483 0.097473 0.019068 -0.030756 0.003464 -0.004049 -0.017734 -0.049131 0.011849 -0.087313 0.018132 0.093301 0.02402 -0.114743 -0.073973 0.022781 -0.003511 0.043055 0.069776 -0.050155 -0.018691 0.078315 -0.016702 0.042535 -0.088182 -0.055301 0.038858 0.051842 0.048028 -0.044823 -0.051279 -0.016467 0.046392 -0.060022 -0.057112 0.130293 0.130172 0.118126 -0.038441 0.104364 0.046031 0.078625 0.021274 -0.007293 -0.002615 -0.127438 -0.014907 -0.071469 0.010509 -0.061879 -0.074339 0.004105 0.010812 0.017881 0.03734 -0.043334 -0.099021 -0.007516 -0.050524 -0.006559 0.035337 0.075333 -0.072907 -0.06585 0.015818 -0.013102 -0.043893 -0.046585 0.030032 -0.080634 0.048787 0.053605 -0.025551 -0.014829 -0.113559 -0.01594 -0.081099 0.11146 0.001189 -0.051934 -0.014657 -0.06457 0.045306 -0.013705 0.02578 0.035293 -0.045513 0.017706 -0.013743 -0.036449 -0.097975 0.00361 0.110656 0.005985 0.003918 -0.083217 0.089543 0.064883 0.063359 -0.067709 0.149407 -0.019442 -0.001442 -0.09393 0.117456 0.079656 -0.028365 -0.054244 0.112843 -0.033993 -0.112373 7.26E-4 -0.05703 -0.04225 0.012609 -0.002963 0.076794 -0.004844 0.072359 -0.110624 0.057226 -0.045802 0.05446 0.03787 0.011126 -0.121707 -0.034402 -0.040453 0.008523 0.012866 -0.061767 -0.146913 -0.032847 0.020639 0.055404 -0.01983 -0.036355 -0.022834 -0.132642 0.007916 -0.089387 -0.003184 -0.019981 -0.097821 -0.026173 0.101657 0.099773 -0.084372 -0.0099 0.020376 0.012375 -0.082308 0.034197 0.003457 0.007728 -0.031867 -0.035034 -0.003641 0.021217 -0.076501";
    QueryInfo query_info(data_str); // groupinfos 对应的 topks
    query_info.MakeAsBuilder();

    RedIndexDocid redindex_docid = 0;
    int ret = builder->Add(redindex_docid, data_str);
    ASSERT_EQ(0, ret);

    const void *dump_content = nullptr;
    size_t dump_size = 0;

    dump_content = builder->Dump(&dump_size);
    ASSERT_TRUE(dump_content != nullptr);
    ASSERT_TRUE(dump_size != 0);

    RedIndex::Pointer loaded_index_p = RedIndexFactory::Load(schema, dump_content, dump_size);
    RedIndex& loaded_index = *loaded_index_p.get();
    //ret = loaded_index.InitForLoad(schema);
    //ASSERT_EQ(0, ret);

    //ret = loaded_index.Load(dump_content, dump_size);
    //ASSERT_EQ(0, ret);

    SlotIndex label = 2591;
    mercury::core::IvfIndex* core_index = dynamic_cast<mercury::core::IvfIndex*>(loaded_index.core_index_.get());
    EXPECT_EQ(label, core_index->GetNearestLabel(query_info.GetVector(), query_info.GetVectorLen()));
    EXPECT_EQ(0, core_index->coarse_index_.search(label).next());
    EXPECT_EQ(label, 
              *(SlotIndex*)core_index->slot_index_profile_.getInfo(redindex_docid));
    //EXPECT_EQ(1, loaded_index.core_index_.current_docid_);
    //EXPECT_EQ(redindex_docid, loaded_index.redindex_docids_[0]);
    EXPECT_EQ(1, core_index->GetDocNum());
    EXPECT_EQ(10001, core_index->GetMaxDocNum());

}

}; // namespace redindex
}; // namespace mercury
