#include <gtest/gtest.h>
#include "framework/index_framework.h"

#define protected public
#define private public
#include "service/ivfpq_searcher.h"
#include "index/ivfpq_index_provider.h"
#include "index/query_distance_matrix.h"
#undef protected
#undef private

using namespace std;

namespace mercury {

static size_t dim = 4UL;
static size_t subDim = 2UL;
static size_t subCentroidNum = 2UL;

class IvfpqSearcherTest : public testing::Test
{
public:
    void SetUp()
    { 
        IndexIvfpq::Pointer 
            ivfpqIndex = IndexIvfpq::Pointer(new IndexIvfpq());

        // index params
        IndexParams::Pointer params = IndexParams::Pointer(new IndexParams); 
        params->set(kBuildDocNumKey, 200);
        params->set(kFeatureInfoSizeKey, sizeof(float)*dim);
        params->set(kDumpDirPathKey, ".");

        IndexStorage::Pointer storage = InstanceFactory::CreateStorage("MMapFileStorage");
        auto file_handle = storage->open("ivfpq_package.index", false);

        ASSERT_TRUE(_provider.init(100, "temp", params));
        ASSERT_TRUE(_provider.load(move(file_handle), ivfpqIndex));
    }

    void TearDown()
    { 
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
        ASSERT_TRUE(index_ivfpq->Dump(storage, file_name, true));

        // created index
        IndexIvfpq::Pointer
        created_index = IndexIvfpq::Pointer(new IndexIvfpq());
        // index params
        params = new IndexParams; 
        params->set(kBuildDocNumKey, 200);
        params->set(kFeatureInfoSizeKey, sizeof(float)*dim);
        params->set(kDumpDirPathKey, ".");
        created_index->set_index_params(params);

        auto file_handle = storage->open("ivfpq.index", false);

        ASSERT_TRUE(created_index->Create(storage, "ivfpq_package.index", move(file_handle)));
        //add doc
        key_t pk = 1;
        docid_t docid = 0;
        vector<float> feature = {1, 1, 1, 1};
        for (; docid < 10; ++docid) {
            EXPECT_EQ(docid, (docid_t)created_index->Add(docid, pk, feature.data(), sizeof(float)*dim));
            ++pk;
        }
    }

    static void TearDownTestCase() 
    {
        system("rm -f ivfpq.index");
        system("rm -f ivfpq_package.index");
    }

public:
    IvfpqIndexProvider _provider;
};

TEST_F(IvfpqSearcherTest, TestInit)
{
    IvfpqSearcher searcher;
    // case 0
    IndexParams params;
    EXPECT_TRUE(searcher.Init(params) == 0);
    //EXPECT_FLOAT_EQ(LEVEL_SCAN_RATIO, searcher._defaultCoarseScanRatio[0]);
    //EXPECT_FLOAT_EQ(COARSE_PROBE_RATIO, searcher._defaultCoarseScanRatio[1]);
    EXPECT_FLOAT_EQ(PQ_SCAN_NUM, searcher._defaultPqScanNum);

    // case 1
    params.set(PARAM_PQ_SEARCHER_COARSE_SCAN_RATIO, "0.05,0.04");
    params.set(PARAM_PQ_SEARCHER_PRODUCT_SCAN_NUM, 1000ul);
    EXPECT_TRUE(searcher.Init(params) == 0);
    //EXPECT_FLOAT_EQ(0.05, searcher._defaultCoarseScanRatio[0]);
    //EXPECT_FLOAT_EQ(0.04, searcher._defaultCoarseScanRatio[1]);
    EXPECT_FLOAT_EQ(1000ul, searcher._defaultPqScanNum);
}

TEST_F(IvfpqSearcherTest, TestLoad)
{
    IvfpqSearcher searcher;
    IndexParams params;
    EXPECT_TRUE(searcher.Init(params) == 0);
    // case 0
    EXPECT_TRUE(searcher.Load(nullptr) != 0);
    // case 1
    IvfpqIndexProvider emptyProvider;
    EXPECT_TRUE(searcher.Load(&emptyProvider) != 0);
    // case 2
    EXPECT_TRUE(searcher.Load(&_provider) == 0);
    //EXPECT_EQ(1ul, searcher._ivfSeekers.size());
}

//TEST_F(IvfpqSearcherTest, TestExhaustiveSearch)
//{
//    IvfpqSearcher searcher;
//    IndexParams params;
//    EXPECT_TRUE(searcher.Init(params) == 0);
//    EXPECT_TRUE(searcher.Load(&_provider) == 0);
//    vector<float> query = {0, 0, 0, 0};
//    GeneralSearchContext context;
//    // case 1
//    EXPECT_TRUE(searcher.ExhaustiveSearch(query.data(), sizeof(float)*dim, 1, &context) == 0);
//    EXPECT_EQ(1ul, context.Result().size());
//    EXPECT_FLOAT_EQ(4.0f, context.Result()[0].score);
//    // case 2
//    context.clean();
//    EXPECT_TRUE(searcher.ExhaustiveSearch(query.data(), sizeof(float)*dim, 20, &context) == 0);
//    EXPECT_EQ(10ul, context.Result().size());
//    EXPECT_FLOAT_EQ(4.0f, context.Result()[0].score);
//}

TEST_F(IvfpqSearcherTest, TestSearch)
{
    IvfpqSearcher searcher;
    IndexParams params;
    EXPECT_TRUE(searcher.Init(params) == 0);
    EXPECT_TRUE(searcher.Load(&_provider) == 0);
    vector<float> query = {0, 0, 0, 0};
    GeneralSearchContext context;
    // case 1
    EXPECT_TRUE(searcher.Search(query.data(), sizeof(float)*dim, 1, &context) == 0);
    EXPECT_EQ(0ul, context.Result().size());
    // case 2
    query = {1, 1, 1, 1};
    context.clean();
    EXPECT_TRUE(searcher.Search(query.data(), sizeof(float)*dim, 10, &context) == 0);
    EXPECT_EQ(10ul, context.Result().size());
}

} // namespace mercury
