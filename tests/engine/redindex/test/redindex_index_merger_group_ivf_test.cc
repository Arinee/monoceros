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
#include "src/core/algorithm/group_ivf/group_ivf_merger.h"
#include "common.h"
#undef protected
#undef private

namespace mercury {
namespace redindex {
using namespace mercury::core;

class RedIndexMergerGroupIvfTest: public testing::Test
{
public:
    void SetUp()
    {
    }

    void TearDown()
    {
    }

};

TEST_F(RedIndexMergerGroupIvfTest, TestMerge) {
  SchemaParams schema = { {"DataType", "float"},
                          {"Method","L2"}, {"Dimension", "256"}, {"IndexType", "GroupIvf"},
                          {"TrainDataPath", "tests/engine/redindex/group_ivf_test_data/"}
                          , {"IvfCoarseScanRatio", "0.5"}, {PARAM_GENERAL_CONTAIN_FEATURE_PROFILE, "true"}};
  RedIndex::Pointer index1 = RedIndexFactory::Create(schema);
  mercury::core::GroupIvfIndex* core_index1 = dynamic_cast<mercury::core::GroupIvfIndex*>(index1->core_index_.get());

  gindex_t group_index = 0;
  SlotIndex label = 1;
  const void *data = core_index1->centroid_resource_manager_.GetCentroidResource(group_index).getValueInRoughMatrix(0, label);
  std::string data_str = DataToStr(data);
  size_t size = core_index1->index_meta_._element_size;
  RedIndexDocid redindex_docid = 0;
  int ret = index1->Add(redindex_docid, data_str);
  ASSERT_EQ(0, ret);
  ASSERT_EQ(1, index1->get_current_doc_num());

  RedIndex::Pointer index2 = RedIndexFactory::Create(schema);
  mercury::core::GroupIvfIndex* core_index2 = dynamic_cast<mercury::core::GroupIvfIndex*>(index2->core_index_.get());
  SlotIndex label2 = 0;
  const void *data2 = core_index2->centroid_resource_manager_.GetCentroidResource(group_index).getValueInRoughMatrix(0, label2);
  std::string data2_str = DataToStr(data2);
  size_t size2 = core_index2->index_meta_._element_size;
  RedIndexDocid redindex_docid2 = 0;
  ret = index2->Add(redindex_docid2, data2_str);
  ASSERT_EQ(0, ret);
  ASSERT_EQ(1, index2->get_current_doc_num());

  std::vector<RedIndex::Pointer> indexes;
  indexes.push_back(index1);
  indexes.push_back(index2);
  std::vector<RedIndexDocid> new_redindex_docids;
  //here we sort by label
  new_redindex_docids.push_back(1);
  new_redindex_docids.push_back(0);

  IndexMergerFactory factory;
  IndexMerger::Pointer index_merger = IndexMergerFactory::Create(schema);

  // ASSERT_EQ(-1, index_merger->PreUpdate(std::vector<RedIndex::Pointer>(), new_redindex_docids));
  bool suc = index_merger->PreUpdate(indexes, new_redindex_docids);
  ASSERT_EQ(1, index_merger->new_redindex_docids_[0]);
  ASSERT_EQ(0, index_merger->new_redindex_docids_[1]);

  suc = index_merger->Merge(indexes);
  ASSERT_EQ(0, suc);

  mercury::core::GroupIvfMerger* group_ivf_merger = dynamic_cast<mercury::core::GroupIvfMerger*>(index_merger->core_merger_.get());
  mercury::core::GroupIvfIndex& merged_index = group_ivf_merger->merged_index_;

  //测试倒排
  ASSERT_EQ(10, merged_index.coarse_index_.getHeader()->slotNum);
  std::vector<uint32_t> coarseIndexLabels;
  coarseIndexLabels.push_back(0);
  coarseIndexLabels.push_back(1);
  for (const auto& e : coarseIndexLabels) {
      mercury::core::CoarseIndex<SmallBlock>::PostingIterator iter = merged_index.coarse_index_.search(e);
      while (!iter.finish()) {
          std::cout<<"label:"<<e<<"; docid:"<<iter.next()<<std::endl;
      }
  }

  mercury::core::CoarseIndex<SmallBlock>::PostingIterator iter = merged_index.coarse_index_.search(0);
  ASSERT_EQ(0, iter.next());

  iter = merged_index.coarse_index_.search(1);
  ASSERT_EQ(1, iter.next());
}

TEST_F(RedIndexMergerGroupIvfTest, TestMergeDelete) {
  SchemaParams schema = { {"DataType", "float"},
                          {"Method","L2"}, {"Dimension", "256"}, {"IndexType", "GroupIvf"},
                          {"TrainDataPath", "tests/engine/redindex/group_ivf_test_data/"}
                          , {"IvfCoarseScanRatio", "0.5"}, {PARAM_GENERAL_CONTAIN_FEATURE_PROFILE, "true"}};
  RedIndex::Pointer index1 = RedIndexFactory::Create(schema);
  mercury::core::GroupIvfIndex* core_index1 = dynamic_cast<mercury::core::GroupIvfIndex*>(index1->core_index_.get());
  
  char *buffer;
  //也可以将buffer作为输出参数
  buffer = getcwd(NULL, 0);
  std::cout << "cwd is:" << buffer << std::endl;
  
  gindex_t group_index = 0;
  SlotIndex label = 1;
  const void *data = core_index1->centroid_resource_manager_.GetCentroidResource(group_index).getValueInRoughMatrix(0, label);
  std::string data_str = DataToStr(data);
  std::string group_str = "1:11||" + data_str;
  size_t size = core_index1->index_meta_._element_size;
  RedIndexDocid redindex_docid = 0;
  int ret = index1->Add(redindex_docid, group_str);
  ASSERT_EQ(0, ret);
  ret = index1->Add(redindex_docid + 1, group_str);
  ASSERT_EQ(0, ret);

  RedIndex::Pointer index2 = RedIndexFactory::Create(schema);
  mercury::core::GroupIvfIndex* core_index2 = dynamic_cast<mercury::core::GroupIvfIndex*>(index2->core_index_.get());
  SlotIndex label2 = 0;
  const void *data2 = core_index2->centroid_resource_manager_.GetCentroidResource(group_index).getValueInRoughMatrix(0, label2);
  group_str = "1:11||" + data_str;
  size_t size2 = core_index2->index_meta_._element_size;
  RedIndexDocid redindex_docid2 = 0;
  ret = index2->Add(redindex_docid2, group_str);
  group_str = "1:11111||" + data_str;
  ret = index2->Add(redindex_docid2 + 1, group_str);
  group_str = "1:11||" + data_str;
  ret = index2->Add(redindex_docid2 + 2, group_str);

  std::vector<RedIndex::Pointer> indexes;
  indexes.push_back(RedIndex::Pointer(index1));
  indexes.push_back(RedIndex::Pointer(index2));
  std::vector<RedIndexDocid> new_redindex_docids;
  //here we sort by label
  new_redindex_docids.push_back(1);
  new_redindex_docids.push_back(2);
  new_redindex_docids.push_back(-1); //mark it as delete
  new_redindex_docids.push_back(0);
  new_redindex_docids.push_back(3);

  IndexMergerFactory factory;
  IndexMerger::Pointer index_merger = IndexMergerFactory::Create(schema);

  // ASSERT_EQ(-1, index_merger->PreUpdate(std::vector<RedIndex::Pointer>(), new_redindex_docids));
  bool suc = index_merger->PreUpdate(indexes, new_redindex_docids);
  ASSERT_EQ(1, index_merger->new_redindex_docids_[0]);
  ASSERT_EQ(-1, index_merger->new_redindex_docids_[2]);

  suc = index_merger->Merge(indexes);
  ASSERT_EQ(0, suc);

  //测试倒排
  mercury::core::GroupIvfMerger* group_ivf_merger = dynamic_cast<mercury::core::GroupIvfMerger*>(index_merger->core_merger_.get());
  mercury::core::GroupIvfIndex& merged_index = group_ivf_merger->merged_index_;

  ASSERT_EQ(10, merged_index.coarse_index_.getHeader()->slotNum);
  std::vector<uint32_t> coarseIndexLabels;
  coarseIndexLabels.push_back(0);
  coarseIndexLabels.push_back(1);
  for (const auto& e : coarseIndexLabels) {
      mercury::core::CoarseIndex<SmallBlock>::PostingIterator iter = merged_index.coarse_index_.search(e);
    while (!iter.finish()) {
      std::cout<<"label:"<<e<<"; docid:"<<iter.next()<<std::endl;
    }
  }

  mercury::core::CoarseIndex<SmallBlock>::PostingIterator iter = merged_index.coarse_index_.search(0);
  ASSERT_TRUE(iter.finish());

  iter = merged_index.coarse_index_.search(6);
  ASSERT_EQ(1, iter.next());
  ASSERT_EQ(2, iter.next());
  ASSERT_EQ(3, iter.next());
}

}; // namespace redindex
}; // namespace mercury
