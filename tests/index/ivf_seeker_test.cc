#include <gtest/gtest.h>
#include "framework/index_framework.h"

#define protected public
#define private public
#include "index/ivf_seeker.h"
#include "index/index_ivfflat.h"
#undef protected
#undef private

using namespace std;

namespace mercury {

static size_t dim = 4UL;

class IvfSeekerTest : public testing::Test
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

TEST_F(IvfSeekerTest, TestInit)
{
    vector<float> coarseScanRatios = {0.05, 0.01};
    IvfSeeker seeker(coarseScanRatios);
    // bad case
    EXPECT_TRUE(seeker.Init(nullptr) != 0); 

    // normal case
    EXPECT_TRUE(seeker.Init(created_index.get()) == 0); 
}

TEST_F(IvfSeekerTest, TestSeek)
{
    //add doc
    //key_t pk = 1;
    //docid_t docid = 0;
    //vector<float> feature = {0.1, 0.1, 0.1, 0.1};
    //EXPECT_EQ(docid, created_index->Add(docid, pk, feature.data(), sizeof(float)*dim));

    //size_t default_nprobe = 1;
    //IvfSeeker seeker(default_nprobe);
    //EXPECT_TRUE(seeker.Init(created_index.get()) == 0); 


    //vector<float> query = {0.1, 0., 0.1, 0.0};
    //GeneralSearchContext context;
    //PostingIterator::Pointer iter = seeker.Seek(query.data(), sizeof(float)*dim, &context);
    //EXPECT_EQ(docid, iter->next());
    //EXPECT_EQ(true, iter->finish());
}

} // namespace mercury
