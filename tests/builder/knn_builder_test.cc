#include "mock_vector_holder.h"
#include <gtest/gtest.h>
#include <bitset>
#include <iostream>

#define protected public
#define private public
#include "index/index_ivfpq.h"
#include "builder/ivfpq_builder.h"
#include "framework/instance_factory.h"
#undef protected
#undef private

using namespace std;

namespace mercury {

class IvfpqBuilderTest : public testing::Test
{
public:
    void SetUp() {}
    void TearDown() 
    {
        // system("rm -rf __temp");
    }

    shared_ptr<IvfpqBuilder> initFloatBuilder(size_t dim = 4, size_t count = 10, bool multiThread = true)
    {
        shared_ptr<IvfpqBuilder> knnPQBuilder(new (nothrow) IvfpqBuilder);
        size_t elemSize = dim * sizeof(float);
        size_t elemCount = count;

        mercury::IndexMeta meta;
        meta.setMeta(mercury::IndexMeta::kTypeFloat, dim);
        meta.setMethod(mercury::IndexDistance::kMethodFloatSquaredEuclidean);
        mercury::IndexParams params;
        params.set(PARAM_GENERAL_BUILDER_MEMORY_QUOTA, 1024L*1024L*1024L);
        params.set(PARAM_GENERAL_BUILDER_THREAD_COUNT, 8);
        if (!multiThread) {
            params.set(PARAM_GENERAL_BUILDER_THREAD_COUNT, 1);
        }
        CentroidResource::Pointer resource(new CentroidResource);
        CentroidResource::RoughMeta roughMeta(elemSize, 1, {10});
        // fragmentNum=2; centroidNum=5
        CentroidResource::IntegrateMeta integrateMeta(elemSize/2, 2, 5);
        EXPECT_TRUE(resource->create(roughMeta, integrateMeta));

        knnPQBuilder->_meta = meta;
        knnPQBuilder->_params = params;
        knnPQBuilder->_resource = resource;
        knnPQBuilder->_segmentDir = "__temp";
        knnPQBuilder->_segment = 0;
        knnPQBuilder->_globalId = 0;

        EXPECT_EQ(0, knnPQBuilder->initProfile(elemCount, elemSize));
        return knnPQBuilder;
    }

    mercury::VectorHolder::Pointer mockHolder()
    {
        MockVectorHolder<float>::Pointer holder(new MockVectorHolder<float>);
        int val = 0;
        for (int i = 0; i < 500; ++i) {
            int32_t key = val + 10;
            vector<float> values;
            for (int j = 0; j < 4; ++j) {
                values.push_back((float)val);
            }
            holder->emplace(key, values);
            val++;
        }
        return holder;
    }

    mercury::VectorHolder::Pointer mockHolderWithLabels()
    {
        MockVectorHolder<float>::Pointer holder(new MockVectorHolder<float>);
        holder->setLabel(true);
        int val = 0;
        for (int i = 0; i < 500; i++) {
            int32_t key = val + 10;
            vector<float> values;
            for (int j = 0; j < 4; ++j) {
                values.push_back((float)val);
            }
            vector<size_t> labels;
            // coarse label
            labels.push_back(1);
            // product label
            labels.push_back(i % 5);
            labels.push_back((i+1) % 5);
            holder->emplace(key, values, labels);
            val++;
        }
        return holder;
    }
};
/*
TEST_F(IvfpqBuilderTest, TestInitResource)
{
    IvfpqBuilder knnPQBuilder;
    mercury::IndexMeta meta;
    meta.setMeta(mercury::IndexMeta::kTypeFloat, 4);
    meta.setMethod(mercury::IndexDistance::kMethodFloatSquaredEuclidean);
    PQCodebook::Pointer pqCodebook(new PQCodebook(meta, 2, 2, 2));
    ASSERT_FALSE(knnPQBuilder.InitResource(meta, pqCodebook));

    float rough[] = {1,2,3,4};
    ASSERT_TRUE(pqCodebook->appendRoughCentroid(rough, sizeof(rough)));
    ASSERT_TRUE(pqCodebook->appendRoughCentroid(rough, sizeof(rough)));
    
    float integrate[] = {1, 2};
    ASSERT_TRUE(pqCodebook->appendIntegrateCentroid(0, integrate, sizeof(integrate)));
    ASSERT_TRUE(pqCodebook->appendIntegrateCentroid(0, integrate, sizeof(integrate)));
    ASSERT_TRUE(pqCodebook->appendIntegrateCentroid(1, integrate, sizeof(integrate)));
    ASSERT_TRUE(pqCodebook->appendIntegrateCentroid(1, integrate, sizeof(integrate)));
    ASSERT_TRUE(knnPQBuilder.InitResource(meta, pqCodebook));
}*/

TEST_F(IvfpqBuilderTest, TestInit)
{
    IvfpqBuilder knnPQBuilder;
    mercury::IndexMeta meta;
    mercury::IndexParams params;
    meta.setMeta(mercury::IndexMeta::kTypeFloat, 4);
    PQCodebook::Pointer pqCodebook(new PQCodebook(meta, 3, 2, 3));
    ASSERT_NE(0, knnPQBuilder.Init(meta, params));
    EXPECT_STREQ("./", knnPQBuilder._segmentDir.substr(0, 2).c_str());

    string path = "__temp";
    params.set(PARAM_PQ_BUILDER_INTERMEDIATE_PATH, path);
    ASSERT_NE(0, knnPQBuilder.Init(meta, params));
    EXPECT_STREQ("__temp/", knnPQBuilder._segmentDir.substr(0, 7).c_str());
    
    params.set(PARAM_PQ_BUILDER_CODEBOOK, pqCodebook);
    ASSERT_NE(0, knnPQBuilder.Init(meta, params));
}

TEST_F(IvfpqBuilderTest, TestCleanup)
{
    IvfpqBuilder knnPQBuilder;
    knnPQBuilder._segmentDir = "__temp/111";
    system(string("mkdir -p __temp/111").c_str());
    string cmd = "touch __temp/111/";
    system(string(cmd + COMPONENT_COARSE_INDEX).c_str());
    system(string(cmd + COMPONENT_PK_PROFILE).c_str());
    system(string(cmd + COMPONENT_PRODUCT_PROFILE).c_str());
    system(string(cmd + COMPONENT_FEATURE_PROFILE).c_str());
    system(string(cmd + COMPONENT_IDMAP).c_str());

    system(string("mkdir __temp/111/segment_0").c_str());
    cmd = "touch __temp/111/segment_0/";
    system(string(cmd + COMPONENT_COARSE_INDEX).c_str());
    system(string(cmd + COMPONENT_PK_PROFILE).c_str());
    system(string(cmd + COMPONENT_PRODUCT_PROFILE).c_str());
    system(string(cmd + COMPONENT_FEATURE_PROFILE).c_str());

    knnPQBuilder._segmentList.push_back("__temp/111/segment_0");
    ASSERT_EQ(0, knnPQBuilder.Cleanup());
}

TEST_F(IvfpqBuilderTest, TestSingleTaskWithoutLabels)
{
    size_t dim = 4;
    auto knnPQBuilder = initFloatBuilder(dim);

    size_t elemSize = dim * sizeof(float);
    float feature[] = {1,2,3,4};
    shared_ptr<char> data(new char[elemSize], std::default_delete<char[]>());
    memcpy(data.get(), reinterpret_cast<const char *>(feature), elemSize);
    knnPQBuilder->singleTaskWithoutLabels(1, data);

    // check CoarseIndex header doc count
    ASSERT_EQ(knnPQBuilder->_featureProfile._header->usedDocNum, 1);
    ASSERT_EQ(knnPQBuilder->_pqcodeProfile._header->usedDocNum, 1);
    ASSERT_EQ(knnPQBuilder->_pkProfile._header->usedDocNum, 1);
}

TEST_F(IvfpqBuilderTest, TestDoSingleBuild)
{
    size_t dim = 4;
    auto knnPQBuilder = initFloatBuilder(dim);

    // init pq build with 2 docs
    size_t elemSize = dim * sizeof(float);
    ASSERT_EQ(0, knnPQBuilder->initProfile(2, elemSize));

    float feature[] = {1,2,3,4};
    shared_ptr<char> data(new char[elemSize], std::default_delete<char[]>());
    memcpy(data.get(), reinterpret_cast<const char *>(feature), elemSize);
    vector<uint16_t> code = {1,2};
    ASSERT_TRUE(knnPQBuilder->doSingleBuild(1, data, 0, code));

    // check CoarseIndex header doc count
    ASSERT_EQ(knnPQBuilder->_featureProfile._header->usedDocNum, 1);
    ASSERT_EQ(knnPQBuilder->_pqcodeProfile._header->usedDocNum, 1);
    ASSERT_EQ(knnPQBuilder->_pkProfile._header->usedDocNum, 1);
    ASSERT_EQ(0, memcmp(knnPQBuilder->_pqcodeProfile.getInfo(0), code.data(), code.size() * sizeof(uint16_t)));

    ASSERT_TRUE(knnPQBuilder->doSingleBuild(1, data, 0, {1,2}));
    ASSERT_TRUE(knnPQBuilder->_featureProfile.isFull());
}

TEST_F(IvfpqBuilderTest, TestFlush)
{
    auto knnPQBuilder = initFloatBuilder();

    ASSERT_TRUE(knnPQBuilder->flushSegment());
    ASSERT_EQ(knnPQBuilder->_segmentList.size(), (size_t)1);
    ASSERT_EQ(knnPQBuilder->_segment, (size_t)1);
}

TEST_F(IvfpqBuilderTest, TestInitProfile)
{
    IvfpqBuilder knnPQBuilder;
    size_t elemSize = 4 * sizeof(float);
    size_t elemCount = 10;

    CentroidResource::Pointer resource(new CentroidResource);
    CentroidResource::RoughMeta roughMeta(elemSize, 1, {10});
    CentroidResource::IntegrateMeta integrateMeta(elemSize/2, 2, 5);
    ASSERT_TRUE(resource->create(roughMeta, integrateMeta));

    knnPQBuilder._resource = resource;

    ASSERT_EQ(0, knnPQBuilder.initProfile(elemCount, elemSize));
    ASSERT_EQ((size_t)knnPQBuilder._coarseIndex.getHeader()->maxDocSize, elemCount);
    ASSERT_EQ((size_t)knnPQBuilder._coarseIndex.getHeader()->capacity, CoarseIndex::calcSize(10, elemCount));
    ASSERT_EQ((size_t)knnPQBuilder._pkProfile.getHeader()->maxDocNum, elemCount);
    ASSERT_EQ((size_t)knnPQBuilder._pkProfile.getHeader()->capacity, ArrayProfile::CalcSize(elemCount, sizeof(uint64_t)));
    ASSERT_EQ((size_t)knnPQBuilder._pqcodeProfile.getHeader()->maxDocNum, elemCount);
    ASSERT_EQ((size_t)knnPQBuilder._pqcodeProfile.getHeader()->capacity, ArrayProfile::CalcSize(elemCount, sizeof(uint16_t) * 2));
    ASSERT_EQ((size_t)knnPQBuilder._featureProfile.getHeader()->maxDocNum, elemCount);
    ASSERT_EQ((size_t)knnPQBuilder._featureProfile.getHeader()->capacity, ArrayProfile::CalcSize(elemCount, elemSize));


    CentroidResource::RoughMeta roughMeta2(elemSize, 2, {100, 10});
    ASSERT_TRUE(resource->create(roughMeta2, integrateMeta));
    knnPQBuilder._resource = resource;
    ASSERT_EQ(0, knnPQBuilder.initProfile(elemCount, elemSize));
    ASSERT_EQ((size_t)knnPQBuilder._coarseIndex.getHeader()->maxDocSize, elemCount);
    ASSERT_EQ((size_t)knnPQBuilder._coarseIndex.getHeader()->capacity, CoarseIndex::calcSize(100*10, elemCount));
}

TEST_F(IvfpqBuilderTest, TestMemQuota2DocCount)
{
    IvfpqBuilder knnPQBuilder;
    mercury::IndexMeta meta;
    meta.setMeta(mercury::VectorHolder::FeatureType::kTypeFloat, 512);
    meta.setMethod(mercury::IndexDistance::kMethodFloatSquaredEuclidean);

    size_t elemSize = 1024;
    //PQCodebook::Pointer pqCodebook(new PQCodebook(meta, 8192, 2048, 64));
    CentroidResource::Pointer resource(new CentroidResource);
    CentroidResource::RoughMeta roughMeta(elemSize, 1, {8192});
    CentroidResource::IntegrateMeta integrateMeta(elemSize/64, 64, 2048);
    bool bret = resource->create(roughMeta, integrateMeta);
    EXPECT_TRUE(bret);

    mercury::IndexParams params;
    //params.set(proxima::paramPQCodeBook, pqCodebook);

    IvfpqBuilder builder;
    builder._resource = resource;

    size_t elemCount = builder.memQuota2DocCount(0, elemSize);
    EXPECT_EQ((size_t)0, elemCount);

    elemCount = builder.memQuota2DocCount(1024L*1024L*1024L, elemSize);
    EXPECT_EQ((size_t)400000, elemCount);
}

TEST_F(IvfpqBuilderTest, TestBuild)
{
    auto knnPQBuilder = initFloatBuilder();

    //prepare holder
    auto holder = mockHolder();
    IndexStorage::Pointer stg = mercury::InstanceFactory::CreateStorage("MMapFileStorage");
    ASSERT_EQ(0, knnPQBuilder->BuildIndex(holder));
    ASSERT_EQ(0, knnPQBuilder->DumpIndex("__temp", stg));
}

TEST_F(IvfpqBuilderTest, TestDump)
{
    auto knnPQBuilder = initFloatBuilder();
    mercury::IndexStorage::Pointer stg = mercury::InstanceFactory::CreateStorage("MMapFileStorage");
    knnPQBuilder->_segmentList = {};
    ASSERT_TRUE(knnPQBuilder->DumpIndex("__temp", stg) < 0);

    // build data
    auto holder = mockHolder();
    // TODO 
    ASSERT_EQ(0, knnPQBuilder->BuildIndex(holder));
    ASSERT_EQ(0, knnPQBuilder->DumpIndex("__temp", stg));
}

TEST_F(IvfpqBuilderTest, TestIsFinished)
{
    auto knnPQBuilder = initFloatBuilder();
    ASSERT_TRUE(!knnPQBuilder->IsFinish());

    //prepare holder
    auto holder = mockHolder();
    IndexStorage::Pointer stg = mercury::InstanceFactory::CreateStorage("MMapFileStorage");
    knnPQBuilder->BuildIndex(holder);
    knnPQBuilder->DumpIndex("__temp", stg);
    ASSERT_TRUE(knnPQBuilder->IsFinish());
}

TEST_F(IvfpqBuilderTest, TestSplitJob)
{
    auto knnPQBuilder = initFloatBuilder();
    auto holder = mockHolder();
    cout << "job split num:" << knnPQBuilder->JobSplit(holder).size() << endl;
}

TEST_F(IvfpqBuilderTest, TestLoadBuildIndex)
{
    IndexIvfpq index_ivfpq;
    IndexStorage::Pointer storage = InstanceFactory::CreateStorage("MMapFileStorage");
    
    auto file_handle = storage->open("__temp/" + PQ_INDEX_FILENAME, false);
    ASSERT_TRUE(index_ivfpq.Load(move(file_handle)));
    
    cout << index_ivfpq.get_centroid_quantizer()->get_centroid_resource()->_roughMatrixSize << endl;
    cout << index_ivfpq.get_centroid_quantizer()->get_centroid_resource()->_integrateMatrixSize 
        << "|" << index_ivfpq.get_centroid_quantizer()->get_centroid_resource()->getIntegrateMeta().elemSize
        << "|" << index_ivfpq.get_centroid_quantizer()->get_centroid_resource()->getIntegrateMeta().fragmentNum
        << "|" << index_ivfpq.get_centroid_quantizer()->get_centroid_resource()->getIntegrateMeta().centroidNum
        << endl;

    cout << (int)index_ivfpq.IsFull() 
         << "|" << index_ivfpq._pFeatureProfile->getHeader()->usedDocNum 
         << "|" << index_ivfpq._pFeatureProfile->getHeader()->maxDocNum 
         << "|" << index_ivfpq._pqcodeProfile->getHeader()->usedDocNum
         << "|" << index_ivfpq._pqcodeProfile->getHeader()->maxDocNum
         << "|" << index_ivfpq._pqcodeProfile->getHeader()->infoSize
         << "|" << index_ivfpq.coarse_index_->getHeader()->slotNum
         << "|" << index_ivfpq.coarse_index_->getHeader()->maxDocSize  
         << "|" << (int)index_ivfpq._pDeleteMap->test(9) 
         << endl;

    for(int index = 0; index < 10; index++){
        cout << "index:" << index << " pk:" << index_ivfpq.getPK(index) << endl;
        const char* datas = (const char*)index_ivfpq.getFeature(index);
        for(int i = 0; i < 4; i++){
            float* fea = (float*)(datas + 4 * i);
            cout << *fea;
        }
        cout << endl;
    }
    /*
    EXPECT_EQ(index_ivfpq.get_centroid_quantizer()->get_centroid_resource()->_roughMatrixSize, size_t((2+2*2)*4));
    EXPECT_FALSE(index_ivfpq.IsFull());
    EXPECT_TRUE(index_ivfpq._pDeleteMap->test(9));
    EXPECT_EQ((size_t)10, index_ivfpq._pFeatureProfile->getHeader()->usedDocNum);
    EXPECT_EQ((size_t)20, index_ivfpq._pFeatureProfile->getHeader()->maxDocNum);

    EXPECT_EQ((size_t)10, index_ivfpq._pqcodeProfile->getHeader()->usedDocNum);
    EXPECT_EQ((size_t)20, index_ivfpq._pqcodeProfile->getHeader()->maxDocNum);
    EXPECT_EQ((size_t)4, index_ivfpq.coarse_index_->getHeader()->slotNum);
    EXPECT_EQ((size_t)20, index_ivfpq.coarse_index_->getHeader()->maxDocSize);
    //valid doc value
    */
}

}
