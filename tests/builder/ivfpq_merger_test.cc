#include "mock_vector_holder.h"
#include <gtest/gtest.h>
#include <bitset>
#include <iostream>

#define protected public
#define private public
#include "index/index_ivfpq.h"
#include "builder/ivfpq_builder.h"
#include "builder/ivfpq_merger.h"
#include "framework/instance_factory.h"
#undef protected
#undef private

using namespace std;

namespace mercury {

class IvfpqMergerTest : public testing::Test
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

    mercury::VectorHolder::Pointer mockHolder(int32_t seed = 10)
    {
        MockVectorHolder<float>::Pointer holder(new MockVectorHolder<float>);
        int val = 0;
        for (int i = 0; i < 1000; ++i) {
            int32_t key = val + seed;
            vector<float> values;
            for (int j = 0; j < 4; ++j) {
                values.push_back((float)val);
            }
            holder->emplace(key, values);
            val++;
        }
        return holder;
    }
};


TEST_F(IvfpqMergerTest, TestNormalMerge) {
    // Build Data 1
    {
        auto knnPQBuilder = initFloatBuilder();
        auto stg = InstanceFactory::CreateStorage("MMapFileStorage");
        // build data
        auto holder = mockHolder(0);
        ASSERT_EQ(0, knnPQBuilder->BuildIndex(holder));
        ASSERT_EQ(0, knnPQBuilder->DumpIndex("__temp1", stg));
    }
    // Build Data 2
    {
        auto knnPQBuilder = initFloatBuilder();
        auto stg = InstanceFactory::CreateStorage("MMapFileStorage");
        // build data
        auto holder = mockHolder(1000);
        ASSERT_EQ(0, knnPQBuilder->BuildIndex(holder));
        ASSERT_EQ(0, knnPQBuilder->DumpIndex("__temp2", stg));
    }
    IvfpqMerger ivfpqMerger;
    IndexParams params;
    ASSERT_EQ(0, ivfpqMerger.Init(params));
    auto stg = InstanceFactory::CreateStorage("MMapFileStorage");
    vector<string> paths = {"__temp1", "__temp2"};
    ASSERT_EQ(0, ivfpqMerger.FeedIndex(paths, stg));
    ASSERT_EQ(0, ivfpqMerger.MergeIndex());
    ASSERT_EQ(0, ivfpqMerger.DumpIndex("__temp3", stg));
    // system("rm -rf __temp1");
    // system("rm -rf __temp2");
    // system("rm -rf __temp3");
}

}
