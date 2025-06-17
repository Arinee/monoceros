#include <gtest/gtest.h>
#include "framework/index_framework.h"

#define protected public
#define private public
#include "index/index_ivfpq.h"
#include "index/query_distance_matrix.h"
#undef protected
#undef private

using namespace std;

namespace mercury {

static size_t dim = 4UL;
static size_t subDim = 2UL;
static size_t subCentroidNum = 2UL;

class PqScorerTest : public testing::Test
{
public:
    void SetUp()
    { 
        created_index = IndexIvfpq::Pointer(new IndexIvfpq());

        // index params
        IndexParams* params = new IndexParams; 
        params->set(kBuildDocNumKey, 20);
        params->set(kFeatureInfoSizeKey, sizeof(float)*dim);
        params->set(kDumpDirPathKey, ".");
        created_index->set_index_params(params);

        IndexStorage::Pointer storage = InstanceFactory::CreateStorage("MMapFileStorage");
        auto file_handle = storage->open("ivfpq.index", false);

        ASSERT_TRUE(created_index->Create(storage, "ivfpq_seg.index", move(file_handle)));
    }

    void TearDown()
    { 
        created_index.reset();
    }

    static void SetUpTestCase() 
    {
        IndexIvfpq::Pointer 
            index_ivfpq = IndexIvfpq::Pointer(new IndexIvfpq);

        // centroid resource
        CentroidResource::Pointer cr(new CentroidResource);
        CentroidResource::RoughMeta rMeta(sizeof(float)*dim, 1, {2});
        CentroidResource::IntegrateMeta iMeta(sizeof(float)*subDim, dim/subDim, subCentroidNum);
        ASSERT_TRUE(cr->create(rMeta, iMeta));
        ASSERT_EQ(cr->_roughMatrixSize, sizeof(float)*dim*2);
        vector<float> centroid0 = {0.0, 0.0, 0.0, 0.0};
        cr->setValueInRoughMatrix(0, 0, centroid0.data());
        vector<float> centroid1 = {1.0, 1.0, 1.0, 1.0};
        cr->setValueInRoughMatrix(0, 1, centroid1.data());
        vector<float> subCentroid0InFragment0 = {0.0, 0.0};
        cr->setValueInIntegrateMatrix(0, 0, subCentroid0InFragment0.data());
        vector<float> subCentroid1InFragment0 = {1.0, 1.0};
        cr->setValueInIntegrateMatrix(0, 1, subCentroid1InFragment0.data());
        vector<float> subCentroid0InFragment1 = {0.0, 0.0};
        cr->setValueInIntegrateMatrix(1, 0, subCentroid0InFragment1.data());
        vector<float> subCentroid1InFragment1 = {1.0, 1.0};
        cr->setValueInIntegrateMatrix(1, 1, subCentroid1InFragment1.data());

        index_ivfpq->get_centroid_quantizer()->set_centroid_resource(cr);

        //index meta
        IndexMeta* index_meta = new IndexMeta;
        index_meta->setType(IndexMeta::kTypeFloat);
        index_meta->setDimension(dim);
        index_meta->setMethod(IndexDistance::kMethodFloatSquaredEuclidean);
        index_ivfpq->set_index_meta(index_meta);

        // index params
        IndexParams* params = new IndexParams; 
        params->set(kBuildDocNumKey, 1000);
        params->set(kFeatureInfoSizeKey, sizeof(float)*dim);
        params->set(kDumpDirPathKey, ".");
        index_ivfpq->set_index_params(params);

        IndexStorage::Pointer storage = InstanceFactory::CreateStorage("MMapFileStorage");
        const string file_name = "ivfpq.index";
        ASSERT_TRUE(index_ivfpq->Dump(storage, file_name , true));
    }  

    static void TearDownTestCase() 
    {}  

public:
    IndexIvfpq::Pointer created_index;
};

//TEST_F(PqScorerTest, TestInit)
//{
//    PqScorer scorer;
//    // bad case
//    EXPECT_TRUE(scorer.Init(nullptr) != 0);
//    // normal case
//    EXPECT_TRUE(scorer.Init(created_index.get()) == 0);
//}
//
//TEST_F(PqScorerTest, TestClone)
//{
//    PqScorer scorer;
//    EXPECT_TRUE(scorer.Init(created_index.get()) == 0);
//    IndexScorer::Pointer clonedScorer = scorer.Clone();
//    PqScorer * pScorer = dynamic_cast<PqScorer*>(clonedScorer.get());
//    EXPECT_TRUE(pScorer != nullptr);
//}
//
//TEST_F(PqScorerTest, TestProcessQuery)
//{
//    PqScorer scorer;
//    EXPECT_TRUE(scorer.Init(created_index.get()) == 0);
//    IndexScorer::Pointer clonedScorer = scorer.Clone();
//    vector<float> query = {0, 0, 0, 0};
//    GeneralSearchContext context;
//    EXPECT_TRUE(clonedScorer->ProcessQuery(query.data(), 0, &context) == 0);
//    EXPECT_TRUE(((PqScorer&)(*clonedScorer)).qdm_.get() != nullptr);
//
//
//    GeneralSearchContext context2 = context;
//    EXPECT_TRUE(clonedScorer->ProcessQuery(query.data(), 0, &context2) == 0);
//    EXPECT_TRUE(((PqScorer&)(*clonedScorer)).qdm_.get() != nullptr);
//}
//
//TEST_F(PqScorerTest, TestScore)
//{
//    //add doc
//    key_t pk = 1;
//    docid_t docid = 0;
//    vector<float> feature = {1, 1, 1, 1};
//    EXPECT_EQ(docid, created_index->Add(docid, pk, feature.data(), sizeof(float)*dim));
//
//    PqScorer scorer;
//    EXPECT_TRUE(scorer.Init(created_index.get()) == 0);
//    IndexScorer::Pointer clonedScorer = scorer.Clone();
//    GeneralSearchContext context;
//    vector<float> query = {0, 0, 0, 0};
//    EXPECT_TRUE(clonedScorer->ProcessQuery(query.data(), 0, &context) == 0);
//    auto qdm = ((PqScorer&)(*clonedScorer)).qdm_;
//    EXPECT_EQ(0, qdm->_distanceArray[0]);
//    EXPECT_EQ(2, qdm->_distanceArray[1]);
//    EXPECT_EQ(0, qdm->_distanceArray[2]);
//    EXPECT_EQ(2, qdm->_distanceArray[3]);
//    EXPECT_FLOAT_EQ(4, clonedScorer->Score(query.data(), 0, docid));
//}

} // namespace mercury
