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
#include "index/ivfflat_index_provider.h"
#undef protected
#undef private

using namespace std;
using namespace mercury;

class IndexProviderTest : public testing::Test
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
        index_meta->setMethod(IndexDistance::kMethodFloatSquaredEuclidean);

        IndexIvfflat index_ivfflat;
        IndexParams* params = new IndexParams; 
        params->set(kBuildDocNumKey, 30);
        params->set(kFeatureInfoSizeKey, 256);
        params->set(kDumpDirPathKey, ".");

        index_ivfflat.set_index_params(params);
        index_ivfflat.set_index_meta(index_meta);
        index_ivfflat.get_centroid_quantizer()->set_centroid_resource(cr);

        IndexStorage::Pointer storage = InstanceFactory::CreateStorage("MMapFileStorage");
        const string file_name = "ivf_flat.index";
        
        ASSERT_TRUE(index_ivfflat.Dump(storage, file_name, true));
        

        auto file_handle = storage->open("ivf_flat.index", false);

        ASSERT_TRUE(index_ivfflat.Create(storage, "ivf_flat_package", move(file_handle)));

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
        for(size_t index = 0; index < 28; index++){
            EXPECT_EQ(index, (size_t)index_ivfflat.Add(index, index, feature.data(), 256));
        }
    }
public:
    
};


TEST_F(IndexProviderTest, TestLoad)
{
    prepareIndexProvider();

    IvfFlatIndexProvider _provider;
    shared_ptr<IndexIvfflat> p_index(new IndexIvfflat);
    IndexStorage::Pointer storage = InstanceFactory::CreateStorage("MMapFileStorage");
    auto file_handle = storage->open("ivf_flat_package", false);
    std::shared_ptr<IndexParams> params(new IndexParams);
    params->set(kBuildDocNumKey, 30);
    params->set(kFeatureInfoSizeKey, 256);
    params->set(kDumpDirPathKey, ".");

    ASSERT_TRUE(_provider.init(100, "temp", params));
    ASSERT_TRUE(_provider.load(move(file_handle),p_index));
    
    EXPECT_EQ((size_t)1, _provider._segments.size());
    //EXPECT_EQ((size_t)28, _provider.getDocNum());
    EXPECT_FALSE(_provider._segments[0]->IsFull());
    EXPECT_EQ(sizeof(float) * 64, _provider._segments[0]->get_index_meta()->sizeofElement());

    IndexIvfflat& index_ivfflat = *((IndexIvfflat*)_provider._segments[0].get());
    EXPECT_EQ(index_ivfflat.get_centroid_quantizer()->get_centroid_resource()->_roughMatrixSize, size_t((2+2*2)*4));
    
    //get doc
    for(size_t index = 0; index < 28; index++){
        EXPECT_EQ(index, _provider.getPK(index));
    }
}

TEST_F(IndexProviderTest, TestUnload)
{
    IvfFlatIndexProvider _provider;
    shared_ptr<IndexIvfflat> p_index(new IndexIvfflat);
    IndexStorage::Pointer storage = InstanceFactory::CreateStorage("MMapFileStorage");
    auto file_handle = storage->open("ivf_flat_package", false);
    std::shared_ptr<IndexParams> params(new IndexParams);
    params->set(kBuildDocNumKey, 30);
    params->set(kFeatureInfoSizeKey, 256);
    params->set(kDumpDirPathKey, ".");

    EXPECT_TRUE(_provider.init(100, "temp", params));
    ASSERT_TRUE(_provider.load(move(file_handle),p_index));

    EXPECT_TRUE(_provider.unload());
    EXPECT_TRUE(_provider._segments.empty());
    //EXPECT_EQ((size_t)1, _provider._segmentIDBegin.size());
    EXPECT_TRUE(_provider._lastSegment == nullptr);
    EXPECT_TRUE(_provider._incrFileHolder.empty());
    //EXPECT_EQ((size_t)0, _provider.getDocNum());
}


TEST_F(IndexProviderTest, TestCreateSegment)
{
    IvfFlatIndexProvider _provider;
    shared_ptr<IndexIvfflat> p_index(new IndexIvfflat);
    IndexStorage::Pointer storage = InstanceFactory::CreateStorage("MMapFileStorage");
    auto file_handle = storage->open("ivf_flat_package", false);
    std::shared_ptr<IndexParams> params(new IndexParams);
    params->set(kBuildDocNumKey, 30);
    params->set(kFeatureInfoSizeKey, 256);
    params->set(kDumpDirPathKey, ".");

    EXPECT_TRUE(_provider.init(100, "temp", params));
    ASSERT_TRUE(_provider.load(move(file_handle), p_index));

    EXPECT_TRUE(_provider._lastSegment != nullptr);
    EXPECT_TRUE(_provider.createSegment(shared_ptr<Index>(new IndexIvfflat)));
    EXPECT_EQ(((IndexIvfflat*)_provider._lastSegment.get())->coarse_index_->getHeader()->slotNum, 4);
    EXPECT_EQ(_provider._lastSegment->_pPKProfile->getHeader()->usedDocNum, 0);
    EXPECT_EQ(_provider._lastSegment->_pPKProfile->getHeader()->maxDocNum, 30);
    EXPECT_TRUE(_provider._lastSegment->_pDeleteMap->testNone());
}

TEST_F(IndexProviderTest, TestGetProfiles)
{
    IvfFlatIndexProvider _provider;
    shared_ptr<IndexIvfflat> p_index(new IndexIvfflat);
    IndexStorage::Pointer storage = InstanceFactory::CreateStorage("MMapFileStorage");
    auto file_handle = storage->open("ivf_flat_package", false);
    std::shared_ptr<IndexParams> params(new IndexParams);
    params->set(kBuildDocNumKey, 30);
    params->set(kFeatureInfoSizeKey, 256);
    params->set(kDumpDirPathKey, ".");

    EXPECT_TRUE(_provider.init(100, "temp", params));
    ASSERT_TRUE(_provider.load(move(file_handle),p_index));

    EXPECT_EQ((uint64_t)2, _provider.getPK(2));
    const void *feature = _provider.getFeature(2);
    vector<float> expectFeature = {11,11,11,11,11,11,11,11,
                             11,11,11,11,11,11,11,11,
                             11,11,11,11,11,11,11,11,
                             11,11,11,11,11,11,11,11,
                             11,11,11,11,11,11,11,11,
                             11,11,11,11,11,11,11,11,
                             11,11,11,11,11,11,11,11,
                             11,11,11,11,11,11,11,11};
    EXPECT_EQ(0, memcmp(feature, expectFeature.data(), sizeof(float) * expectFeature.size()));
}

TEST_F(IndexProviderTest, TestAddDelete)
{
    IvfFlatIndexProvider _provider;
    shared_ptr<IndexIvfflat> p_index(new IndexIvfflat);
    IndexStorage::Pointer storage = InstanceFactory::CreateStorage("MMapFileStorage");
    auto file_handle = storage->open("ivf_flat_package", false);
    std::shared_ptr<IndexParams> params(new IndexParams);
    params->set(kBuildDocNumKey, 30);
    params->set(kFeatureInfoSizeKey, 256);
    params->set(kDumpDirPathKey, ".");

    EXPECT_TRUE(_provider.init(100, "temp", params));
    ASSERT_TRUE(_provider.load(move(file_handle),p_index));
    
    cout <<_provider._segments[0]->_pPKProfile->getDocNum() << endl;
    cout <<_provider._segments[0]->_pFeatureProfile->getDocNum() << endl;
    //ADD
    vector<float> feature = {11,11,11,11,11,11,11,11,
                             11,11,11,11,11,11,11,11,
                             11,11,11,11,11,11,11,11,
                             11,11,11,11,11,11,11,11,
                             11,11,11,11,11,11,11,11,
                             11,11,11,11,11,11,11,11,
                             11,11,11,11,11,11,11,11,
                             11,11,11,11,11,11,11,11};

    EXPECT_FALSE(_provider._segments[0]->IsFull());
    EXPECT_EQ(28, (int)_provider.addVector(28, feature.data(), 0));
    EXPECT_EQ(29, (int)_provider.addVector(29, feature.data(), 0));
    //DELETE
    //EXPECT_TRUE(_provider.deleteVector(11));
    //EXPECT_TRUE(_provider._segments[0]->_pDeleteMap->test(11));   
    //FULL TEST
    EXPECT_TRUE(_provider._segments.size() == 1);
    EXPECT_TRUE(_provider._segments[0]->IsFull());
}

//AUTO CREATE SEGMENT1 TEST
TEST_F(IndexProviderTest, TestAddWhenCreate)
{
    IvfFlatIndexProvider _provider;
    shared_ptr<IndexIvfflat> p_index(new IndexIvfflat);
    IndexStorage::Pointer storage = InstanceFactory::CreateStorage("MMapFileStorage");
    auto file_handle = storage->open("ivf_flat_package", false);
    std::shared_ptr<IndexParams> params(new IndexParams);
    params->set(kBuildDocNumKey, 30);
    params->set(kFeatureInfoSizeKey, 256);
    params->set(kDumpDirPathKey, ".");

    EXPECT_TRUE(_provider.init(100, "temp", params));
    ASSERT_TRUE(_provider.load(move(file_handle),p_index));
    
    
    //ADD
    vector<float> feature = {11,11,11,11,11,11,11,11,
                             11,11,11,11,11,11,11,11,
                             11,11,11,11,11,11,11,11,
                             11,11,11,11,11,11,11,11,
                             11,11,11,11,11,11,11,11,
                             11,11,11,11,11,11,11,11,
                             11,11,11,11,11,11,11,11,
                             11,11,11,11,11,11,11,11};

    //FULL TEST
    EXPECT_TRUE(_provider._segments[0]->IsFull());
    //ADD
    EXPECT_EQ(GET_GLOID(1, 0), _provider.addVector(30, feature.data(), 0));
    EXPECT_EQ(GET_GLOID(1, 1), _provider.addVector(31, feature.data(), 0));
    EXPECT_EQ(2, (int)_provider._segments[1]->_pPKProfile->getDocNum());
    //DELETE
    EXPECT_FALSE(_provider._segments[1]->_pDeleteMap->test(0));
    EXPECT_TRUE(_provider.deleteVector(30));
    EXPECT_TRUE(_provider._segments[1]->_pDeleteMap->test(0));
}

TEST_F(IndexProviderTest, TestLoadSegNew)
{
    IvfFlatIndexProvider _provider;
    shared_ptr<IndexIvfflat> p_index(new IndexIvfflat);
    IndexStorage::Pointer storage = InstanceFactory::CreateStorage("MMapFileStorage");
    auto file_handle = storage->open("testdata/segment_1548086897", false);
    std::shared_ptr<IndexParams> params(new IndexParams);
    params->set(kBuildDocNumKey, 30);
    params->set(kFeatureInfoSizeKey, 256);
    params->set(kDumpDirPathKey, ".");
    
    ASSERT_TRUE(file_handle != nullptr);
    ASSERT_TRUE(_provider.init(100, "temp", params));
    ASSERT_TRUE(_provider.load(move(file_handle),p_index));
    
    EXPECT_EQ((size_t)1, _provider._segments.size());
    //EXPECT_EQ((size_t)2, _provider.getDocNum());
    EXPECT_EQ((size_t)2, _provider._segments[0]->_pPKProfile->getDocNum());
    EXPECT_TRUE(!_provider._segments[0]->IsFull());
    EXPECT_EQ(sizeof(float) * 64, _provider._segments[0]->get_index_meta()->sizeofElement());

    IndexIvfflat& index_ivfflat = *((IndexIvfflat*)_provider._segments[0].get());
    EXPECT_EQ(index_ivfflat.get_centroid_quantizer()->get_centroid_resource()->_roughMatrixSize, size_t((2+2*2)*4));
    
    //get doc
    for(size_t index = 0; index < 1; index++){
        EXPECT_EQ(30, (int)index_ivfflat.getPK(index));
    }
}
