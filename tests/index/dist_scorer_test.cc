#include <gtest/gtest.h>
#include "framework/index_framework.h"

#define protected public
#define private public
#include "index/dist_scorer.h"
#include "index/index_ivfflat.h"
#undef protected
#undef private

using namespace std;

namespace mercury {

static size_t dim = 4UL;

class DistScorerTest : public testing::Test
{
public:
    void SetUp()
    { 
        created_index = IndexIvfflat::Pointer(new IndexIvfflat());

        // index params
        IndexParams* params = new IndexParams; 
        params->set(kBuildDocNumKey, 20);
        params->set(kFeatureInfoSizeKey, sizeof(float)*dim);
        params->set(kDumpDirPathKey, ".");
        created_index->set_index_params(params);

        IndexStorage::Pointer storage = InstanceFactory::CreateStorage("MMapFileStorage");
        auto file_handle = storage->open("ivf_flat.index", false);

        ASSERT_TRUE(created_index->Create(storage, "ivf_flat_seg.index", move(file_handle)));
    }

    void TearDown()
    { 
        created_index.reset();
    }

    static void SetUpTestCase() 
    {
        IndexIvfflat::Pointer 
            index_ivfflat = IndexIvfflat::Pointer(new IndexIvfflat);

        // centroid resource
        CentroidResource::Pointer cr(new CentroidResource);
        CentroidResource::RoughMeta rMeta(sizeof(float)*dim, 1, {2});
        CentroidResource::IntegrateMeta iMeta(0, 0, 0);
        ASSERT_TRUE(cr->create(rMeta, iMeta));
        ASSERT_EQ(cr->_roughMatrixSize, sizeof(float)*dim*2);
        vector<float> centroid0 = {0.0, 0.0, 0.0, 0.0};
        cr->setValueInRoughMatrix(0, 0, centroid0.data());
        vector<float> centroid1 = {1.0, 1.0, 1.0, 1.0};
        cr->setValueInRoughMatrix(0, 1, centroid1.data());
        index_ivfflat->get_centroid_quantizer()->set_centroid_resource(cr);

        //index meta
        IndexMeta* index_meta = new IndexMeta;
        index_meta->setType(IndexMeta::kTypeFloat);
        index_meta->setDimension(dim);
        index_meta->setMethod(IndexDistance::kMethodFloatSquaredEuclidean);
        index_ivfflat->set_index_meta(index_meta);

        // index params
        IndexParams* params = new IndexParams; 
        params->set(kBuildDocNumKey, 1000);
        params->set(kFeatureInfoSizeKey, sizeof(float)*dim);
        params->set(kDumpDirPathKey, ".");
        index_ivfflat->set_index_params(params);

        IndexStorage::Pointer storage = InstanceFactory::CreateStorage("MMapFileStorage");
        const string file_name = "ivf_flat.index";
        ASSERT_TRUE(index_ivfflat->Dump(storage, file_name , true));
    }  

    static void TearDownTestCase() 
    {}  

public:
    IndexIvfflat::Pointer created_index;
};

TEST_F(DistScorerTest, TestInit)
{
    DistScorer scorer;
    // bad case
    EXPECT_TRUE(scorer.Init(nullptr) != 0);
    // normal case
    EXPECT_TRUE(scorer.Init(created_index.get()) == 0);
}

TEST_F(DistScorerTest, TestClone)
{
    DistScorer scorer;
    EXPECT_TRUE(scorer.Init(created_index.get()) == 0);
    IndexScorer::Pointer clonedScorer = scorer.Clone();
    DistScorer * distScorer = dynamic_cast<DistScorer*>(clonedScorer.get());
    EXPECT_TRUE(distScorer != nullptr);
}

TEST_F(DistScorerTest, TestProcessQuery)
{
    DistScorer scorer;
    EXPECT_TRUE(scorer.Init(created_index.get()) == 0);
    { // empty case
        IndexScorer::Pointer clonedScorer = scorer.Clone();
        GeneralSearchContext context;
        EXPECT_TRUE(clonedScorer->ProcessQuery(nullptr, 0, &context) == 0);
        EXPECT_TRUE(((DistScorer&)(*clonedScorer)).index_meta_.method() == IndexDistance::kMethodFloatSquaredEuclidean);
    }

    { // replace case
        IndexScorer::Pointer clonedScorer = scorer.Clone();
        GeneralSearchContext context;
        context.setSearchMethod(IndexDistance::kMethodFloatNormalizedEuclidean);
        EXPECT_TRUE(clonedScorer->ProcessQuery(nullptr, 0, &context) == 0);
        EXPECT_TRUE(((DistScorer&)(*clonedScorer)).index_meta_.method() == IndexDistance::kMethodFloatNormalizedEuclidean);
    }
}

TEST_F(DistScorerTest, TestScore)
{
    //add doc
    key_t pk = 1;
    docid_t docid = 0;
    vector<float> feature = {1, 1, 1, 1};
    EXPECT_EQ(docid, (docid_t)created_index->Add(docid, pk, feature.data(), sizeof(float)*dim));

    DistScorer scorer;
    EXPECT_TRUE(scorer.Init(created_index.get()) == 0);
    IndexScorer::Pointer clonedScorer = scorer.Clone();
    GeneralSearchContext context;
    EXPECT_TRUE(clonedScorer->ProcessQuery(nullptr, 0, &context) == 0);
    {
        vector<float> query = {0, 0, 0, 0};
        EXPECT_FLOAT_EQ(4, clonedScorer->Score(query.data(), 0, docid));
    }
    {
        vector<float> query = {1, 0, 0, 0};
        EXPECT_FLOAT_EQ(3, clonedScorer->Score(query.data(), 0, docid));
    }

}

} // namespace mercury
