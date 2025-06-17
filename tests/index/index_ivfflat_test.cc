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
#include "index/index_ivfflat.h"
#undef protected
#undef private

using namespace std;
using namespace mercury;

class IvfFlatIndexTest : public testing::Test
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
        CentroidResource::IntegrateMeta iMeta(0, 0, 0);
        ASSERT_TRUE(cr->create(rMeta, iMeta));
        ASSERT_EQ(cr->_roughMatrixSize, size_t((2+2*2)*4));

        int32_t val = 1;
        cr->setValueInRoughMatrix(0, 0, (char*)&val);
        val++;
        cr->setValueInRoughMatrix(0, 1, (char*)&val);
        val++;
        cr->setValueInRoughMatrix(1, 0, (char*)&val);
        val++;
        cr->setValueInRoughMatrix(1, 1, (char*)&val);

        //index meta
        IndexMeta* index_meta = new IndexMeta;
        index_meta->setType(IndexMeta::kTypeFloat);
        index_meta->setDimension(64);
        index_meta->setMethod(mercury::IndexDistance::kMethodFloatSquaredEuclidean);

        IndexIvfflat index_ivfflat;
        IndexParams* params = new IndexParams; 
        params->set(kBuildDocNumKey, 1000);
        params->set(kFeatureInfoSizeKey, 256);
        params->set(kDumpDirPathKey, ".");

        index_ivfflat.set_index_params(params);
        index_ivfflat.set_index_meta(index_meta);
        index_ivfflat.get_centroid_quantizer()->set_centroid_resource(cr);

        IndexStorage::Pointer storage = InstanceFactory::CreateStorage("MMapFileStorage");
        const string file_name = "ivf_flat.index";
        
        ASSERT_TRUE(index_ivfflat.Dump(storage, file_name , true));
    }

public:

};

TEST_F(IvfFlatIndexTest, TestCreate)
{
    prepareIndexProvider();
    IndexIvfflat index_ivfflat;
    
    IndexParams* params = new IndexParams; 
    params->set(kBuildDocNumKey, 20);
    params->set(kFeatureInfoSizeKey, 256);
    params->set(kDumpDirPathKey, ".");

    index_ivfflat.set_index_params(params);

    IndexStorage::Pointer storage = InstanceFactory::CreateStorage("MMapFileStorage");
    auto file_handle = storage->open("ivf_flat.index", false);

    ASSERT_TRUE(index_ivfflat.Create(storage, "ivf_flat_seg.index", move(file_handle)));

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
    for(int index = 0; index < 10; index++){
        EXPECT_EQ(index, index_ivfflat.Add(index, index, feature.data(), 256));
    }
    ASSERT_TRUE(index_ivfflat.RemoveId(9));
    EXPECT_TRUE(index_ivfflat._pDeleteMap->test(9));
}

TEST_F(IvfFlatIndexTest, TestLoad)
{
    IndexIvfflat index_ivfflat;
    IndexStorage::Pointer storage = InstanceFactory::CreateStorage("MMapFileStorage");
    
    auto file_handle = storage->open("ivf_flat_seg.index", false);
    ASSERT_TRUE(index_ivfflat.Load(move(file_handle)));

    EXPECT_EQ(index_ivfflat.get_centroid_quantizer()->get_centroid_resource()->_roughMatrixSize, size_t((2+2*2)*4));
    EXPECT_FALSE(index_ivfflat.IsFull());
    EXPECT_TRUE(index_ivfflat._pDeleteMap->test(9));
    EXPECT_EQ((long)10, index_ivfflat._pFeatureProfile->getHeader()->usedDocNum);
    EXPECT_EQ((long)20, index_ivfflat._pFeatureProfile->getHeader()->maxDocNum);

    EXPECT_EQ((long)4, index_ivfflat.coarse_index_->getHeader()->slotNum);
    EXPECT_EQ((long)20, index_ivfflat.coarse_index_->getHeader()->maxDocSize);
    //valid doc value
    for(size_t index = 0; index < 10; index++){
        EXPECT_EQ(index, index_ivfflat.getPK(index));
    }
}

TEST_F(IvfFlatIndexTest, TestRemove)
{
    IndexIvfflat index_ivfflat;
    IndexStorage::Pointer storage = InstanceFactory::CreateStorage("MMapFileStorage");
    
    auto file_handle = storage->open("ivf_flat_seg.index", false);
    ASSERT_TRUE(index_ivfflat.Load(move(file_handle)));

    EXPECT_FALSE(index_ivfflat._pDeleteMap->test(6));
    EXPECT_TRUE(index_ivfflat.RemoveId(6));
    EXPECT_TRUE(index_ivfflat._pDeleteMap->test(6));
}

TEST_F(IvfFlatIndexTest, TestDump)
{
    IndexIvfflat index_ivfflat;
    IndexStorage::Pointer storage = InstanceFactory::CreateStorage("MMapFileStorage");
    auto file_handle = storage->open("ivf_flat_seg.index", false);
    ASSERT_TRUE(index_ivfflat.Load(move(file_handle)));
    IndexParams* params = new IndexParams; 
    params->set(kBuildDocNumKey, 20);
    params->set(kFeatureInfoSizeKey, 256);
    params->set(kDumpDirPathKey, ".");
    index_ivfflat.set_index_params(params);

    EXPECT_TRUE(index_ivfflat._pDeleteMap->test(9));
    EXPECT_TRUE(index_ivfflat._pDeleteMap->test(6));
    EXPECT_TRUE(index_ivfflat.Dump(storage, "ivf_flat_seg_dump_meta_only.index", true));
    EXPECT_TRUE(index_ivfflat.Dump(storage, "ivf_flat_seg_dump.index", false));
}

TEST_F(IvfFlatIndexTest, TestLoadFromDump)
{
    IndexIvfflat index_ivfflat;
    IndexStorage::Pointer storage = InstanceFactory::CreateStorage("MMapFileStorage");
    
    auto file_handle = storage->open("ivf_flat_seg_dump.index", false);
    ASSERT_TRUE(index_ivfflat.Load(move(file_handle)));

    EXPECT_EQ(index_ivfflat.get_centroid_quantizer()->get_centroid_resource()->_roughMatrixSize, size_t((2+2*2)*4));
    EXPECT_FALSE(index_ivfflat.IsFull());
    EXPECT_TRUE(index_ivfflat._pDeleteMap->test(6));
    EXPECT_EQ((long)10, index_ivfflat._pFeatureProfile->getHeader()->usedDocNum);
    EXPECT_EQ((long)20, index_ivfflat._pFeatureProfile->getHeader()->maxDocNum);

    EXPECT_EQ((long)4, index_ivfflat.coarse_index_->getHeader()->slotNum);
    EXPECT_EQ((long)20, index_ivfflat.coarse_index_->getHeader()->maxDocSize);
    //valid doc value
    for(size_t index = 0; index < 10; index++){
        EXPECT_EQ(index, index_ivfflat.getPK(index));
    }
}

//TEST_F(IvfFlatIndexTest, TestSearchAndCalc)
//{
//    IndexIvfflat index_ivfflat;
//    IndexStorage::Pointer storage = InstanceFactory::CreateStorage("MMapFileStorage");
//    auto file_handle = storage->open("ivf_flat_seg.index", false);
//    ASSERT_TRUE(index_ivfflat.Load(move(file_handle)));
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
//    IvfPostingIterator ivf_iterator = index_ivfflat.SearchtTopNPosting(feature.data(), 256, 5, context);
//    
//    cout << ivf_iterator.ivf_posting_list_.size()  << "|" << ivf_iterator.finish() << endl;
//
//    while(!ivf_iterator.finish())
//    {
//        docid_t doc_id = ivf_iterator.next();
//        cout << "get doc:" << doc_id << endl;
//        EXPECT_EQ(doc_id, index_ivfflat.getPK(doc_id));
//        EXPECT_EQ(0, memcmp(index_ivfflat.getFeature(doc_id), feature_vec.data(), sizeof(float) * feature_vec.size()));
//        cout << "score:" << index_ivfflat.CalcDistance(feature.data(), 256, doc_id) << endl;
//        //EXPECT_EQ(feature_vec.data(), index_ivfflat.getFeature(doc_id));
//    }
//
//}

TEST_F(IvfFlatIndexTest, TestAdd)
{
    IndexIvfflat index_ivfflat;
    IndexStorage::Pointer storage = InstanceFactory::CreateStorage("MMapFileStorage");
    
    auto file_handle = storage->open("ivf_flat_seg.index", false);
    ASSERT_TRUE(index_ivfflat.Load(move(file_handle)));

    
    vector<float> feature = {   11,11,11,11,11,11,11,11,
                                11,11,11,11,11,11,11,11,
                                11,11,11,11,11,11,11,11,
                                11,11,11,11,11,11,11,11,
                                11,11,11,11,11,11,11,11,
                                11,11,11,11,11,11,11,11,
                                11,11,11,11,11,11,11,11,
                                11,11,11,11,11,11,11,11};
                             
    //add doc
    for(int index = 10; index < 20; index++){
        EXPECT_EQ(index, index_ivfflat.Add(index, index, feature.data(), 256));
    }
}

TEST_F(IvfFlatIndexTest, TestLoadFull)
{
    IndexIvfflat index_ivfflat;
    IndexStorage::Pointer storage = InstanceFactory::CreateStorage("MMapFileStorage");
    
    auto file_handle = storage->open("ivf_flat_seg.index", false);
    ASSERT_TRUE(index_ivfflat.Load(move(file_handle)));

    EXPECT_EQ(index_ivfflat.get_centroid_quantizer()->get_centroid_resource()->_roughMatrixSize, size_t((2+2*2)*4));
    EXPECT_TRUE(index_ivfflat.IsFull());
    EXPECT_EQ((long)20, index_ivfflat._pFeatureProfile->getHeader()->usedDocNum);
    EXPECT_EQ((long)20, index_ivfflat._pFeatureProfile->getHeader()->maxDocNum);

    //valid doc value
    for(size_t index = 0; index < 20; index++){
        EXPECT_EQ(index, index_ivfflat.getPK(index));
    }
}
