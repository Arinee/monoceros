#include <gtest/gtest.h>
#include "framework/index_package.h"
#include "framework/index_meta.h"
#include "framework/index_distance.h"
#include "framework/vector_holder.h"

#define protected public
#define private public
#include "index/coarse_index.h"
#include "index/array_profile.h"
#include "index/centroid_resource.h"
#include "index/index_ivfpq.h"
#undef protected
#undef private

using namespace std;
using namespace mercury;

class IvfpqIndexTest : public testing::Test
{
public:
    void SetUp()
    {

    }

    void TearDown()
    {

    }

    void prepareIndexProvider()
    {
        //make meta package
        CentroidResource::Pointer cr(new CentroidResource);
        CentroidResource::RoughMeta rMeta(4, 2, {2, 2});
        CentroidResource::IntegrateMeta iMeta(128, 2, 256);
        ASSERT_TRUE(cr->create(rMeta, iMeta));
        ASSERT_EQ(cr->_roughMatrixSize, size_t((2+2*2)*4));
        ASSERT_EQ(cr->_integrateMatrixSize, size_t(128*2*256));

        int32_t val = 1;
        cr->setValueInRoughMatrix(0, 0, (char*)&val);
        val++;
        cr->setValueInRoughMatrix(0, 1, (char*)&val);
        val++;
        cr->setValueInRoughMatrix(1, 0, (char*)&val);
        val++;
        cr->setValueInRoughMatrix(1, 1, (char*)&val);

        int val2[32];
        memset(val2, 1, 128);

        for(size_t m = 0; m < 256; m++){
            cr->setValueInIntegrateMatrix(0, m, (char*)val2);
            cr->setValueInIntegrateMatrix(1, m, (char*)val2);
        }
        
        //index meta
        IndexMeta* index_meta = new IndexMeta;
        index_meta->setType(IndexMeta::kTypeFloat);
        index_meta->setDimension(64);
        index_meta->setMethod(mercury::IndexDistance::kMethodFloatSquaredEuclidean);

        IndexIvfpq index_ivfpq;
        IndexParams* params = new IndexParams; 
        params->set(kBuildDocNumKey, 1000);
        params->set(kFeatureInfoSizeKey, 256);
        params->set(kDumpDirPathKey, ".");

        index_ivfpq.set_index_params(params);
        index_ivfpq.set_index_meta(index_meta);
        index_ivfpq.get_centroid_quantizer()->set_centroid_resource(cr);

        IndexStorage::Pointer storage = InstanceFactory::CreateStorage("MMapFileStorage");
        const string file_name = "ivfpq.index";
        
        ASSERT_TRUE(index_ivfpq.Dump(storage, file_name , true));
    }
public:
};

TEST_F(IvfpqIndexTest, TestCreate)
{
    prepareIndexProvider();
    IndexIvfpq index_ivfpq;
    
    IndexParams* params = new IndexParams; 
    params->set(kBuildDocNumKey, 20);
    params->set(kFeatureInfoSizeKey, 256);
    params->set(kDumpDirPathKey, ".");

    index_ivfpq.set_index_params(params);

    IndexStorage::Pointer storage = InstanceFactory::CreateStorage("MMapFileStorage");
    auto file_handle = storage->open("ivfpq.index", false);

    ASSERT_TRUE(index_ivfpq.Create(storage, "ivfpq_seg.index", move(file_handle)));

    // add some data
    vector<float> feature = {11,11,11,11,11,11,11,11,
                             11,11,11,11,11,11,11,11,
                             11,11,11,11,11,11,11,11,
                             11,11,11,11,11,11,11,11,
                             11,11,11,11,11,11,11,11,
                             11,11,11,11,11,11,11,11,
                             11,11,11,11,11,11,11,11,
                             11,11,11,11,11,11,11,11};
                             
    //add doc
    for(size_t index = 0; index < 10; index++){
        EXPECT_EQ(index, (size_t)index_ivfpq.Add(index, index, feature.data(), 256));
    }
    ASSERT_TRUE(index_ivfpq.RemoveId(9));
    EXPECT_TRUE(index_ivfpq._pDeleteMap->test(9));
}

TEST_F(IvfpqIndexTest, TestLoad)
{
    IndexIvfpq index_ivfpq;
    IndexStorage::Pointer storage = InstanceFactory::CreateStorage("MMapFileStorage");
    
    auto file_handle = storage->open("ivfpq_seg.index", false);
    ASSERT_TRUE(index_ivfpq.Load(move(file_handle)));

    EXPECT_EQ(index_ivfpq.get_centroid_quantizer()->get_centroid_resource()->_roughMatrixSize, size_t((2+2*2)*4));
    EXPECT_FALSE(index_ivfpq.IsFull());
    EXPECT_TRUE(index_ivfpq._pDeleteMap->test(9));
    EXPECT_EQ(10, (int)index_ivfpq._pFeatureProfile->getHeader()->usedDocNum);
    EXPECT_EQ(20, (int)index_ivfpq._pFeatureProfile->getHeader()->maxDocNum);

    EXPECT_EQ(10, (int)index_ivfpq._pqcodeProfile->getHeader()->usedDocNum);
    EXPECT_EQ(20, (int)index_ivfpq._pqcodeProfile->getHeader()->maxDocNum);
    EXPECT_EQ(4, (int)index_ivfpq.coarse_index_->getHeader()->slotNum);
    EXPECT_EQ(20, (int)index_ivfpq.coarse_index_->getHeader()->maxDocSize);
    //valid doc value
    for(size_t index = 0; index < 10; index++){
        EXPECT_EQ(index, index_ivfpq.getPK(index));
    }
}

TEST_F(IvfpqIndexTest, TestRemove)
{
    IndexIvfpq index_ivfpq;
    IndexStorage::Pointer storage = InstanceFactory::CreateStorage("MMapFileStorage");
    
    auto file_handle = storage->open("ivfpq_seg.index", false);
    ASSERT_TRUE(index_ivfpq.Load(move(file_handle)));

    EXPECT_FALSE(index_ivfpq._pDeleteMap->test(6));
    EXPECT_TRUE(index_ivfpq.RemoveId(6));
    EXPECT_TRUE(index_ivfpq._pDeleteMap->test(6));
}

TEST_F(IvfpqIndexTest, TestDump)
{
    IndexIvfpq index_ivfpq;
    IndexStorage::Pointer storage = InstanceFactory::CreateStorage("MMapFileStorage");
    auto file_handle = storage->open("ivfpq_seg.index", false);
    ASSERT_TRUE(index_ivfpq.Load(move(file_handle)));
    IndexParams* params = new IndexParams; 
    params->set(kBuildDocNumKey, 20);
    params->set(kFeatureInfoSizeKey, 256);
    params->set(kDumpDirPathKey, ".");
    index_ivfpq.set_index_params(params);

    EXPECT_TRUE(index_ivfpq._pDeleteMap->test(9));
    EXPECT_TRUE(index_ivfpq._pDeleteMap->test(6));
    EXPECT_TRUE(index_ivfpq.Dump(storage, "ivf_flat_seg_dump_meta_only.index", true));
    EXPECT_TRUE(index_ivfpq.Dump(storage, "ivf_flat_seg_dump.index", false));
}

TEST_F(IvfpqIndexTest, TestLoadFromDump)
{
    IndexIvfpq index_ivfpq;
    IndexStorage::Pointer storage = InstanceFactory::CreateStorage("MMapFileStorage");
    
    auto file_handle = storage->open("ivfpq_seg.index", false);
    ASSERT_TRUE(index_ivfpq.Load(move(file_handle)));

    EXPECT_EQ(index_ivfpq.get_centroid_quantizer()->get_centroid_resource()->_roughMatrixSize, size_t((2+2*2)*4));
    EXPECT_FALSE(index_ivfpq.IsFull());
    EXPECT_TRUE(index_ivfpq._pDeleteMap->test(6));
    EXPECT_EQ(10, (int)index_ivfpq._pFeatureProfile->getHeader()->usedDocNum);
    EXPECT_EQ(20, (int)index_ivfpq._pFeatureProfile->getHeader()->maxDocNum);

    EXPECT_EQ(4, (int)index_ivfpq.coarse_index_->getHeader()->slotNum);
    EXPECT_EQ(20, (int)index_ivfpq.coarse_index_->getHeader()->maxDocSize);
    //valid doc value
    for(size_t index = 0; index < 10; index++){
        EXPECT_EQ(index, index_ivfpq.getPK(index));
    }
}

//TEST_F(IvfpqIndexTest, TestSearchAndCalc)
//{
//    IndexIvfpq index_ivfpq;
//    IndexStorage::Pointer storage = InstanceFactory::CreateStorage("MMapFileStorage");
//    auto file_handle = storage->open("ivfpq_seg.index", false);
//    ASSERT_TRUE(index_ivfpq.Load(move(file_handle)));
//
//    vector<float> feature = { 1,11,1,11,11,11,11,1,
//                              1,11,11,11,11,1,11,11,
//                              1,11,1,11,11,11,11,11,
//                              1,11,11,11,1,11,11,11,
//                              1,11,11,11,11,11,11,11,
//                              1,11,11,11,1,1,1,11,
//                              1,11,1,11,11,11,11,11,
//                              1,11,11,11,1,11,11,11 };
//
//    GeneralSearchContext* context = new GeneralSearchContext;
//    IvfPostingIterator ivf_iterator = index_ivfpq.SearchtTopNPosting(feature.data(), 256, 5, context);
//
//    cout << ivf_iterator.ivf_posting_list_.size()  << "|" << ivf_iterator.finish() << endl;
//}

//TEST_F(IvfpqIndexTest, TestQDM)
//{
//    IndexIvfpq index_ivfpq;
//    IndexStorage::Pointer storage = InstanceFactory::CreateStorage("MMapFileStorage");
//    auto file_handle = storage->open("ivfpq_seg.index", false);
//    ASSERT_TRUE(index_ivfpq.Load(move(file_handle)));
//
//    vector<float> feature = { 1,11,1,11,11,11,11,1,
//                              1,11,11,11,11,1,11,11,
//                              1,11,1,11,11,11,11,11,
//                              1,11,11,11,1,11,11,11,
//                              1,11,11,11,11,11,11,11,
//                              1,11,11,11,1,1,1,11,
//                              1,11,1,11,11,11,11,11,
//                              1,11,11,11,1,11,11,11 };
//
//    vector<float> feature_vec = {11,11,11,11,11,11,11,11,
//                                11,11,11,11,11,11,11,11,
//                                11,11,11,11,11,11,11,11,
//                                11,11,11,11,11,11,11,11,
//                                11,11,11,11,11,11,11,11,
//                                11,11,11,11,11,11,11,11,
//                                11,11,11,11,11,11,11,11,
//                                11,11,11,11,11,11,11,11};
//
//    GeneralSearchContext* context = new GeneralSearchContext;
//    IvfPostingIterator ivf_iterator = index_ivfpq.SearchtTopNPosting(feature.data(), 256, 5, context);
//
//    cout << ivf_iterator.ivf_posting_list_.size()  << "|" << ivf_iterator.finish() << endl;
//    GeneralSearchContext* pq_context = new GeneralSearchContext();
//    QueryDistanceMatrix::Pointer qdm = index_ivfpq.InitQueryDistanceMatrix(feature.data(), pq_context);
//    EXPECT_TRUE(qdm.get() != nullptr);
//
//    while(!ivf_iterator.finish())
//    {
//        docid_t doc_id = ivf_iterator.next();
//        //cout << "get doc:" << doc_id << endl;
//
//        //const uint16_t *productInfo = reinterpret_cast<const uint16_t *>(index_ivfpq._pqcodeProfile->getInfo(doc_id));
//        //for(size_t i = 0; i < 2; i++){
//        //   cout << productInfo[i] << endl;
//        //}
//        //break;
//        EXPECT_EQ(doc_id, index_ivfpq.getPK(doc_id));
//        EXPECT_EQ(0, memcmp(index_ivfpq.getFeature(doc_id), feature_vec.data(), sizeof(float) * feature_vec.size()));
//        cout << "score:" << index_ivfpq.CalcDistance(doc_id, qdm.get()) << endl;
//    }
//}

TEST_F(IvfpqIndexTest, TestAdd)
{
    IndexIvfpq index_ivfpq;
    IndexStorage::Pointer storage = InstanceFactory::CreateStorage("MMapFileStorage");
 
    auto file_handle = storage->open("ivfpq_seg.index", false);
    ASSERT_TRUE(index_ivfpq.Load(move(file_handle)));

 
    vector<float> feature = {   11,11,11,11,11,11,11,11,
                                11,11,11,11,11,11,11,11,
                                11,11,11,11,11,11,11,11,
                                11,11,11,11,11,11,11,11,
                                11,11,11,11,11,11,11,11,
                                11,11,11,11,11,11,11,11,
                                11,11,11,11,11,11,11,11,
                                11,11,11,11,11,11,11,11};
                          
    //add doc
    for(size_t index = 10; index < 20; index++){
        EXPECT_EQ(index, (size_t)index_ivfpq.Add(index, index, feature.data(), 256));
    }
}

TEST_F(IvfpqIndexTest, TestLoadFull)
{
    IndexIvfpq index_ivfpq;
    IndexStorage::Pointer storage = InstanceFactory::CreateStorage("MMapFileStorage");
 
    auto file_handle = storage->open("ivfpq_seg.index", false);
    ASSERT_TRUE(index_ivfpq.Load(move(file_handle)));

    EXPECT_EQ(index_ivfpq.get_centroid_quantizer()->get_centroid_resource()->_roughMatrixSize, size_t((2+2*2)*4));
    EXPECT_TRUE(index_ivfpq.IsFull());
    EXPECT_EQ(20, (int)index_ivfpq._pFeatureProfile->getHeader()->usedDocNum);
    EXPECT_EQ(20, (int)index_ivfpq._pFeatureProfile->getHeader()->maxDocNum);

    //valid doc value
    for(size_t index = 0; index < 20; index++){
        EXPECT_EQ(index, (size_t)index_ivfpq.getPK(index));
    }
}
