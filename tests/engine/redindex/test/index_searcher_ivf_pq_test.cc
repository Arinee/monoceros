/// Copyright (c) 2019, xiaohongshu Inc. All rights reserved.
/// Author: kailuo <kailuo@xiaohongshu.com>
/// Created: 2019-09-06 00:59

#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
//#include <sys/types.h>
//#include <sys/stat.h>
//#include <unistd.h>

#define protected public
#define private public
#include "src/engine/redindex/index_searcher_factory.h"
#include "src/engine/redindex/index_merger_factory.h"
#include "src/engine/redindex/redindex_factory.h"
#include "src/engine/redindex/redindex_index.h"
#include "src/engine/redindex/index_merger.h"
#include "src/engine/redindex/index_searcher.h"
#include "src/core/algorithm/ivf_pq/ivf_pq_index.h"
#undef protected
#undef private

namespace mercury {
namespace redindex {
using namespace mercury::core;

class IndexSearcherIvfPQTest: public testing::Test
{
public:
    void SetUp()
    {
    }

    void TearDown()
    {
    }

};

TEST_F(IndexSearcherIvfPQTest, TestSearch) {
    char *buffer;
    //也可以将buffer作为输出参数
    buffer = getcwd(NULL, 0);
    std::cout << "cwd is:" << buffer << std::endl;
  SchemaParams schema = { {"DataType", "float"},
                          {"Method","L2"}, {"Dimension", "256"}, {"IndexType", "IvfPQ"},
                          {"TrainDataPath", "tests/engine/redindex/test_data/"}
                          , {"IvfCoarseScanRatio", "1"}
                          , {"IvfPQScanNum", "100"}};
  RedIndex::Pointer index1 = RedIndexFactory::Create(schema);
  mercury::core::IvfPqIndex* core_index1 = dynamic_cast<mercury::core::IvfPqIndex*>(index1->core_index_.get());

  SlotIndex label = 1;
  const void *data = core_index1->centroid_resource_.getValueInRoughMatrix(0, label);
  std::vector<float> float_vec;
  float_vec.resize(256);
  memcpy(float_vec.data(), data, 256 * sizeof(float));

  std::string data_str = StringUtil::vectorToStr(float_vec);
  size_t size = core_index1->index_meta_.sizeofElement();

  RedIndexDocid redindex_docid = 0;
  int ret = index1->Add(redindex_docid, data_str);
  ret = index1->Add(redindex_docid + 1, data_str);

  RedIndex::Pointer index2 = RedIndexFactory::Create(schema);
  mercury::core::IvfPqIndex* core_index2 = dynamic_cast<mercury::core::IvfPqIndex*>(index2->core_index_.get());

  ASSERT_EQ(0, ret);
  SlotIndex label2 = 0;
  const void *data2 = core_index2->centroid_resource_.getValueInRoughMatrix(0, label2);
  memcpy(float_vec.data(), data2, 256 * sizeof(float));

  std::string data2_str = StringUtil::vectorToStr(float_vec);

  size_t size2 = core_index2->index_meta_.sizeofElement();
  RedIndexDocid redindex_docid2 = 0;
  ret = index2->Add(redindex_docid2, data_str);
  ret = index2->Add(redindex_docid2 + 1, data2_str);

  std::vector<RedIndex::Pointer> indexes;
  indexes.push_back(index1);
  indexes.push_back(index2);
  std::vector<RedIndexDocid> new_redindex_docids;
  new_redindex_docids.push_back(1);
  new_redindex_docids.push_back(2);
  new_redindex_docids.push_back(-1); //mark it as delete
  new_redindex_docids.push_back(0);

  IndexMergerFactory factory;
  IndexMerger::Pointer index_merger = IndexMergerFactory::Create(schema);
  ASSERT_TRUE(index_merger != nullptr);

  ASSERT_EQ(-1, index_merger->PreUpdate(std::vector<RedIndex::Pointer>(), new_redindex_docids));
  bool suc = index_merger->PreUpdate(indexes, new_redindex_docids);
  ASSERT_EQ(1, index_merger->new_redindex_docids_.at(0));
  ASSERT_EQ(-1, index_merger->new_redindex_docids_.at(2));

  suc = index_merger->Merge(indexes);
  ASSERT_EQ(0, suc);

  IndexSearcherFactory searcher_factory;
  IndexSearcher::Pointer index_searcher = searcher_factory.Create(schema);
  size_t dump_size;
  const void* dump_data = index_merger->Dump(&dump_size);
  int ret_code = index_searcher->LoadIndex(dump_data, dump_size);
  ASSERT_EQ(ret_code, 0);

  IndexParams index_params;
  GeneralSearchContext context(index_params);
  QueryInfo query_info(data_str);
  ASSERT_TRUE(query_info.MakeAsSearcher());
  index_searcher->Search(query_info, &context);

  ASSERT_EQ(ret_code, 0);
  ASSERT_EQ(3, context.Result().size());
  ASSERT_EQ(0, context.Result().at(0).gloid);
  ASSERT_EQ(1, context.Result().at(1).gloid);
  ASSERT_EQ(2, context.Result().at(2).gloid);
}

}; // namespace redindex
}; // namespace mercury
