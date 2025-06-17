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
#include "src/engine/redindex/index_merger.h"
#include "src/engine/redindex/index_merger_factory.h"
#include "src/engine/redindex/redindex_factory.h"
#include "src/core/algorithm/ivf_pq/ivf_pq_index.h"
#include "src/core/algorithm/ivf_pq/ivf_pq_merger.h"
#include "common.h"
#undef protected
#undef private

namespace mercury {
namespace redindex {
using namespace mercury::core;

class RedIndexMergerIvfPQTest: public testing::Test
{
public:
    void SetUp()
    {
    }

    void TearDown()
    {
    }

};

TEST_F(RedIndexMergerIvfPQTest, TestMerge) {
  SchemaParams schema = { {"DataType", "float"},
                          {"Method","L2"}, {"Dimension", "256"}, {"IndexType", "IvfPQ"},
                          {"TrainDataPath", "tests/engine/redindex/test_data/"}
                          , {"IvfCoarseScanRatio", "0.5"}
                          , {"IvfPQScanNum", "100"}};
  RedIndex::Pointer index1 = RedIndexFactory::Create(schema);
  mercury::core::IvfPqIndex* core_index1 = dynamic_cast<mercury::core::IvfPqIndex*>(index1->core_index_.get());

  SlotIndex label = 1;
  const void *data = core_index1->centroid_resource_.getValueInRoughMatrix(0, label);
  std::string data_str = DataToStr(data);
  size_t size = core_index1->index_meta_.sizeofElement();
  RedIndexDocid redindex_docid = 0;
  int ret = index1->Add(redindex_docid, data_str);
  ASSERT_EQ(0, ret);
  ret = index1->Add(redindex_docid + 1, data_str);

  RedIndex::Pointer index2 = RedIndexFactory::Create(schema);
  mercury::core::IvfPqIndex* core_index2 = dynamic_cast<mercury::core::IvfPqIndex*>(index2->core_index_.get());
  SlotIndex label2 = 0;
  const void *data2 = core_index2->centroid_resource_.getValueInRoughMatrix(0, label2);
  std::string data2_str = DataToStr(data2);
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
  IndexMerger::Pointer index_merger = factory.Create(schema);

  ASSERT_EQ(-1, index_merger->PreUpdate(std::vector<RedIndex::Pointer>(), new_redindex_docids));
  bool suc = index_merger->PreUpdate(indexes, new_redindex_docids);
  ASSERT_EQ(1, index_merger->new_redindex_docids_[0]);
  ASSERT_EQ(-1, index_merger->new_redindex_docids_[2]);

  suc = index_merger->Merge(indexes);
  ASSERT_EQ(0, suc);

  mercury::core::IvfPqMerger* core_merger = dynamic_cast<mercury::core::IvfPqMerger*>(index_merger->core_merger_.get());
  mercury::core::IvfPqIndex& merged_index = core_merger->merged_index_;

  //测试倒排
  ASSERT_EQ(2, merged_index.coarse_index_.getHeader()->slotNum);
  std::vector<uint32_t> coarseIndexLabels;
  coarseIndexLabels.push_back(0);
  coarseIndexLabels.push_back(1);
  for (const auto& e : coarseIndexLabels) {
      mercury::core::CoarseIndex<BigBlock>::PostingIterator iter = merged_index.coarse_index_.search(e);
    while (!iter.finish()) {
      std::cout<<"pqlabel:"<<e<<"; docid:"<<iter.next()<<std::endl;
    }
  }

  mercury::core::CoarseIndex<BigBlock>::PostingIterator iter = merged_index.coarse_index_.search(0);
  ASSERT_EQ(0, iter.next());

  iter = merged_index.coarse_index_.search(1);
  ASSERT_EQ(1, iter.next());
  ASSERT_EQ(2, iter.next());

  //测试slot_index_profile
  ASSERT_EQ(0, *(SlotIndex*)merged_index.GetSlotIndexProfile().getInfo(0));
  ASSERT_EQ(1, *(SlotIndex*)merged_index.GetSlotIndexProfile().getInfo(1));
  ASSERT_EQ(1, *(SlotIndex*)merged_index.GetSlotIndexProfile().getInfo(2));

  //测试get_pqcode_profile
  uint16_t* ex_profile = (uint16_t*)core_index1->GetPqCodeProfile().getInfo(0);
  uint16_t* real_profile = (uint16_t*)merged_index.GetPqCodeProfile().getInfo(1);
  EXPECT_EQ(ex_profile[0], real_profile[0]);
  EXPECT_EQ(ex_profile[31], real_profile[31]);

  ex_profile = (uint16_t*)core_index2->GetPqCodeProfile().getInfo(1);
  real_profile = (uint16_t*)merged_index.GetPqCodeProfile().getInfo(0);
  EXPECT_EQ(ex_profile[0], real_profile[0]);
  EXPECT_EQ(ex_profile[31], real_profile[31]);
}

}; // namespace redindex
}; // namespace mercury
