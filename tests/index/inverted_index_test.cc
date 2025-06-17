#include <gtest/gtest.h>
#include "framework/index_package.h"
#include "framework/index_meta.h"
#include "framework/index_distance.h"
#include "framework/vector_holder.h"

#define protected public
#define private public
#include "index/mult_cat_index.h"
#undef protected
#undef private

using namespace std;
using namespace mercury;

class InvertedIndex : public testing::Test
{
public:
    void SetUp()
    {

    }

    void TearDown()
    {

    }
};

TEST_F(InvertedIndex, TestLoad)
{
    IndexParams index_params;
    MultCatIndex multCatIndex(index_params);
    IndexStorage::Pointer storage = InstanceFactory::CreateStorage("MMapFileStorage");
    auto file_handle = storage->open("testdata/catinverted.indexes", false);
    
    ASSERT_TRUE(multCatIndex.Load(move(file_handle)));
    EXPECT_EQ(200, (int)multCatIndex.GetIDMap()->count());
    //Traversing
    size_t traverNum = multCatIndex.GetIDMap()->count();
    for(size_t index = 0; index < traverNum; index++){
        MultCatIndex::CateFeeder feeder = multCatIndex.GetCateFeeder(index);
        uint64_t docId;
        int count = 0;

        while((docId = feeder.GetNextDoc()) != INVALID_DOCID){
            count++;
            //std::cout << docId << " ";    
        }
        EXPECT_EQ(50, count);
        //std::cout << std::endl << count;
    }
}

TEST_F(InvertedIndex, TestDump)
{
    IndexParams index_params;
    index_params.set(kDumpDirPathKey, ".");

    MultCatIndex multCatIndex(index_params);
    IndexStorage::Pointer storage = InstanceFactory::CreateStorage("MMapFileStorage");
    auto file_handle = storage->open("testdata/catinverted.indexes", false);
    
    ASSERT_TRUE(multCatIndex.Load(move(file_handle)));
    ASSERT_TRUE(multCatIndex.Dump(storage, "dump_catinverted_test.indexes"));
}
