#include "index/centroid_resource.h"
#include "framework/utility/time_helper.h"
#include <vector>
#include <gtest/gtest.h>

#define protected public
#define private public
#include "index/query_distance_matrix.h"
#undef protected
#undef private


using namespace std;
using namespace mercury;

std::vector<int16_t> rawRoughMatrix = {
#include "rough_matrix.txt.h"
};


class QueryDistanceMatrixTest : public testing::Test
{
public:
    void SetUp()
    {
        /*
        FILE *fp = fopen("testdata/integrate_matrix.txt", "r");
        ASSERT_TRUE(fp != nullptr);
        char buf[4096];
        while (fgets(buf, 4096, fp)) {
            std::string tmp(buf);
            std::vector<std::string> vec = StringUtil::split(tmp, std::string(","), true);
            ASSERT_EQ(8UL+1, vec.size());
            for (size_t i = 0, sz = vec.size() - 1; i < sz; ++i) {
                _rawIntegrateMatrix.push_back(atof(vec[i].c_str()));
            }
        }
        fclose(fp);
        ASSERT_EQ(2048UL*64*8, _rawIntegrateMatrix.size());
        ASSERT_FLOAT_EQ(2.560858, _rawIntegrateMatrix[_rawIntegrateMatrix.size()-1]);

        CentroidResource::Pointer centroidResource 
            = CentroidResource::Pointer(new CentroidResource());
        CentroidResource::RoughMeta roughMeta;
        roughMeta._centroidNum = 8192U;
        roughMeta._fragmentNum = 64U;
        CentroidResource::IntegrateMeta integrateMeta;
        integrateMeta._centroidNum = 2048U;
        integrateMeta._fragmentNum = 64U;
        integrateMeta._subDimension = 8;
        integrateMeta._subDimensionCost = 8;
        ASSERT_TRUE(centroidResource->create(roughMeta, integrateMeta));
        for (size_t centroid = 0; centroid < roughMeta._centroidNum; ++centroid) {
            for (size_t fragmentIndex = 0; fragmentIndex < roughMeta._fragmentNum; ++fragmentIndex) {
                int16_t value = rawRoughMatrix[centroid * roughMeta._fragmentNum + fragmentIndex];
                centroidResource->setValueInRoughMatrix(centroid, fragmentIndex, value);
            }
        }
        for (size_t fragmentIndex = 0; fragmentIndex < integrateMeta._fragmentNum; ++fragmentIndex) {
            for (size_t centroid = 0; centroid < integrateMeta._centroidNum; ++centroid) {
                size_t index = (fragmentIndex * integrateMeta._centroidNum * integrateMeta._subDimensionCost)
                    + (centroid * integrateMeta._subDimensionCost);
                std::vector<float> value(&_rawIntegrateMatrix[index], &_rawIntegrateMatrix[index + integrateMeta._subDimensionCost]);
                centroidResource->setValueInIntegrateMatrix(fragmentIndex, centroid, value);
            }
        }
        EXPECT_EQ(_rawIntegrateMatrix[3*2048*8 + 4*8], centroidResource->getValueInIntegrateMatrix(3, 4)[0]);
        for (size_t centroid = 0; centroid < integrateMeta._centroidNum; ++centroid) {
            for (size_t fragmentIndex = 0; fragmentIndex < integrateMeta._fragmentNum; ++fragmentIndex) {
                size_t index = (fragmentIndex * integrateMeta._centroidNum * integrateMeta._subDimensionCost)
                    + (centroid * integrateMeta._subDimensionCost);
                for (size_t dim = 0; dim < integrateMeta._subDimensionCost; ++dim) {
                    _integrateMatrixCentroidFrist.push_back(_rawIntegrateMatrix[index + dim]);
                }
            }
        }
        EXPECT_EQ(_rawIntegrateMatrix[3*2048*8 + 4*8], _integrateMatrixCentroidFrist[4*64*8 + 3*8]);

        IndexMeta indexMeta;
        indexMeta.setDimension(512);
        indexMeta.setType(VectorHolder::kTypeFloat);
        indexMeta.setMethod(IndexDistance::kMethodFloatSquaredEuclidean);

        _qdm.setCentroidResource(centroidResource);
        _qdm.setIndexMeta(indexMeta);
        */
    }

    void TearDown()
    {
    }
protected:
    IndexMeta _indexMeta;
    std::vector<float> _rawIntegrateMatrix;
    std::vector<float> _integrateMatrixCentroidFrist;
    QueryDistanceMatrix _qdm;
};

TEST_F(QueryDistanceMatrixTest, TestMemoryWriteSeq)
{
    _qdm._distanceArray = new(nothrow) distance_t[_qdm._centroidNum * _qdm._fragmentNum];
    ASSERT_TRUE(_qdm._distanceArray);
    // begin to compute 
    uint64_t t11 = Monotime::MicroSeconds();
    for (size_t fragmentIndex = 0; fragmentIndex < _qdm._fragmentNum ; ++fragmentIndex) {
        for (size_t centroidIndex = 0; centroidIndex < _qdm._centroidNum; ++ centroidIndex) {
            size_t index = fragmentIndex * _qdm._centroidNum + centroidIndex;
            _qdm._distanceArray[index] = 100.0f;
        }
    }

    fprintf(stdout, "cost: %lu.\n", Monotime::MicroSeconds() - t11);
}

TEST_F(QueryDistanceMatrixTest, TestMemoryWriteInterval)
{
    _qdm._distanceArray = new(nothrow) distance_t[_qdm._centroidNum * _qdm._fragmentNum];
    ASSERT_TRUE(_qdm._distanceArray);
    // begin to compute 
    uint64_t t11 = Monotime::MicroSeconds();
    for (size_t fragmentIndex = 0; fragmentIndex < _qdm._fragmentNum ; ++fragmentIndex) {
        for (size_t centroidIndex = 0; centroidIndex < _qdm._centroidNum; ++ centroidIndex) {
            size_t index = centroidIndex * _qdm._fragmentNum + fragmentIndex;
            _qdm._distanceArray[index] = 100.0f;
        }
    }

    fprintf(stdout, "cost: %lu.\n", Monotime::MicroSeconds() - t11);
}


TEST_F(QueryDistanceMatrixTest, TestComputeDistanceMatrix)
{
    return;
    std::vector<float> query = {
#include "query_data.txt.h"
    };
    EXPECT_EQ(512UL, query.size());

    // begin to compute 
    uint64_t t11 = Monotime::MicroSeconds();
    EXPECT_TRUE(_qdm.computeDistanceMatrix(query.data()));
    //_qdm._distanceArray = new(nothrow) distance_t[_qdm._centroidNum * _qdm._fragmentNum];
    //ASSERT_TRUE(_qdm._distanceArray);
    //for (size_t fragmentIndex = 0; fragmentIndex < _qdm._fragmentNum ; ++fragmentIndex) {
    //    const feature_t* fragmentFeature = &query[fragmentIndex*_qdm._subDimension];
    //    for (size_t centroidIndex = 0; centroidIndex < _qdm._centroidNum; ++ centroidIndex) {
    //        feature_t* fragmentCenter = _qdm._centroidResource->getValueInIntegrateMatrix(fragmentIndex, centroidIndex);
    //        size_t index = fragmentIndex * _qdm._centroidNum + centroidIndex;
    //        _qdm._distanceArray[index] = _qdm._indexMeta.distance(fragmentFeature, fragmentCenter);
    //    }
    //}

    fprintf(stdout, "cost: %lu.\n", Monotime::MicroSeconds() - t11);
    fprintf(stdout, "result: ");
    for (size_t centroid= 0; centroid < 10; ++centroid) {
        size_t fragment = 3;
        fprintf(stdout, "%f, ", _qdm.getDistance(centroid, fragment));
    }
    fprintf(stdout, "\n");
}

TEST_F(QueryDistanceMatrixTest, TestComputeDistanceMatrixOld)
{
    std::vector<float> query = {
#include "query_data.txt.h"
    };
    EXPECT_EQ(512UL, query.size());

    // begin to compute 
    uint64_t t11 = Monotime::MicroSeconds();
    _qdm._distanceArray = new(nothrow) distance_t[_qdm._centroidNum * _qdm._fragmentNum];
    ASSERT_TRUE(_qdm._distanceArray);
    for (size_t fragmentIndex = 0; fragmentIndex < _qdm._fragmentNum ; ++fragmentIndex) {
        const feature_t* fragmentFeature = &query[_qdm._elemSize];
        for (size_t centroidIndex = 0; centroidIndex < _qdm._centroidNum; ++ centroidIndex) {
            const void* fragmentCenter = _qdm._centroidResource->getValueInIntegrateMatrix(fragmentIndex, centroidIndex);
            size_t index = centroidIndex * _qdm._fragmentNum + fragmentIndex;
            // _distanceArray的长度为64*2048，实际是将二维矩阵用vector表示
            // 行代表和第N(0<=N<=2047)个中心点，列代表第M(0<=M<=63)个fragment分段
            _qdm._distanceArray[index] = _qdm._indexMeta.distance(fragmentFeature, fragmentCenter);
        }
    }
    fprintf(stdout, "cost: %lu.\n", Monotime::MicroSeconds() - t11);
    fprintf(stdout, "result: ");
    for (size_t centroid= 0; centroid < 10; ++centroid) {
        size_t fragment = 3;
        fprintf(stdout, "%f, ", _qdm._distanceArray[centroid * _qdm._fragmentNum + fragment]);
    }
    fprintf(stdout, "\n");
}

TEST_F(QueryDistanceMatrixTest, TestComputeDistanceMatrixExperiment)
{
    std::vector<float> query = {
#include "query_data.txt.h"
    };
    EXPECT_EQ(512UL, query.size());

    // begin to compute 
    uint64_t t11 = Monotime::MicroSeconds();
    _qdm._distanceArray = new(nothrow) distance_t[_qdm._centroidNum * _qdm._fragmentNum];
    ASSERT_TRUE(_qdm._distanceArray);
    for (size_t centroidIndex = 0; centroidIndex < _qdm._centroidNum; ++ centroidIndex) {
        for (size_t fragmentIndex = 0; fragmentIndex < _qdm._fragmentNum ; ++fragmentIndex) {
            const feature_t* fragmentFeature = &query[_qdm._elemSize];
            const void* fragmentCenter = 0;//TODO //&_integrateMatrixCentroidFrist[centroidIndex * _qdm._fragmentNum * _qdm._subDimension + fragmentIndex * _qdm._subDimension];
            size_t index = fragmentIndex * _qdm._centroidNum + centroidIndex;
            _qdm._distanceArray[index] = _qdm._indexMeta.distance(fragmentFeature, fragmentCenter);
        }
    }
    fprintf(stdout, "cost: %lu.\n", Monotime::MicroSeconds() - t11);
    fprintf(stdout, "result: ");
    for (size_t centroid= 0; centroid < 10; ++centroid) {
        size_t fragment = 3;
        fprintf(stdout, "%f, ", _qdm._distanceArray[fragment * _qdm._centroidNum + centroid]);
    }
    fprintf(stdout, "\n");
}


TEST_F(QueryDistanceMatrixTest, TestTableDistanceCentroidAsRow)
{
    size_t roughCentroidNum = 8192UL;
    size_t integrateCentroidNum = 2048UL;
    size_t fragmentNum = 64UL;
    //size_t subDimension = 8UL;
    EXPECT_EQ(roughCentroidNum * fragmentNum, rawRoughMatrix.size());
    // prepare data
    std::vector<int16_t> roughMatrix = rawRoughMatrix;
    std::vector<float> qdmDistance;
    qdmDistance.resize(fragmentNum*integrateCentroidNum);
    for (size_t fragmentIndex = 0; fragmentIndex < fragmentNum; ++fragmentIndex) {
        for (size_t centroid = 0; centroid < integrateCentroidNum; ++centroid) {
            // make up data
            qdmDistance[fragmentIndex * integrateCentroidNum + centroid] = fragmentIndex + centroid;
        }
    }
    std::vector<DistNode> result;
    for (size_t i = 0; i < roughCentroidNum; ++i) {
        result.emplace_back(i, 0.0f);
    }

    // begin to compute 
    uint64_t t11 = Monotime::MicroSeconds();
    for (uint32_t roughCentroid = 0; roughCentroid < roughCentroidNum; ++roughCentroid) {
        float dist = 0.0f;
        for (size_t fragmentIndex = 0; fragmentIndex <fragmentNum; ++fragmentIndex) {
            size_t integrateCentroid = roughMatrix[roughCentroid * fragmentNum + fragmentIndex];
            dist += qdmDistance[fragmentIndex * integrateCentroidNum + integrateCentroid];
        }
        result[roughCentroid].dist = dist;
    }
    fprintf(stdout, "cost: %lu, result: ", Monotime::MicroSeconds() - t11);
    for (size_t roughCentroid = 0; roughCentroid < roughCentroidNum; ++roughCentroid) {
        if (roughCentroid % 1024 == 0) {
            fprintf(stdout, "%f, ", result[roughCentroid].dist);
        }
    }
    fprintf(stdout, "\n");
}

TEST_F(QueryDistanceMatrixTest, TestTableDistanceCentroidAsRowAndAlsoInDistanceTable)
{
    size_t roughCentroidNum = 8192UL;
    size_t integrateCentroidNum = 2048UL;
    size_t fragmentNum = 64UL;
    //size_t subDimension = 8UL;
    EXPECT_EQ(roughCentroidNum * fragmentNum, rawRoughMatrix.size());
    // prepare data
    std::vector<int16_t> roughMatrix = rawRoughMatrix;
    std::vector<float> qdmDistance;
    qdmDistance.resize(fragmentNum*integrateCentroidNum);
    for (size_t centroid = 0; centroid < integrateCentroidNum; ++centroid) {
        for (size_t fragmentIndex = 0; fragmentIndex < fragmentNum; ++fragmentIndex) {
            // make up data
            qdmDistance[centroid * fragmentNum + fragmentIndex] = fragmentIndex + centroid;
        }
    }
    std::vector<DistNode> result;
    for (size_t i = 0; i < roughCentroidNum; ++i) {
        result.emplace_back(i, 0.0f);
    }

    // begin to compute 
    uint64_t t11 = Monotime::MicroSeconds();
    for (uint32_t roughCentroid = 0; roughCentroid < roughCentroidNum; ++roughCentroid) {
        float dist = 0.0f;
        for (size_t fragmentIndex = 0; fragmentIndex <fragmentNum; ++fragmentIndex) {
            size_t integrateCentroid = roughMatrix[roughCentroid * fragmentNum + fragmentIndex];
            dist += qdmDistance[integrateCentroid * fragmentNum + fragmentIndex];
        }
        result[roughCentroid].dist = dist;
    }
    fprintf(stdout, "cost: %lu, result: ", Monotime::MicroSeconds() - t11);
    for (size_t roughCentroid = 0; roughCentroid < roughCentroidNum; ++roughCentroid) {
        if (roughCentroid % 1024 == 0) {
            fprintf(stdout, "%f, ", result[roughCentroid].dist);
        }
    }
    fprintf(stdout, "\n");
}


TEST_F(QueryDistanceMatrixTest, TestTableDistanceImmitateProfileRotate)
{
    size_t roughCentroidNum = 8192UL;
    size_t integrateCentroidNum = 2048UL;
    size_t fragmentNum = 64UL;
    //size_t subDimension = 8UL;
    EXPECT_EQ(roughCentroidNum * fragmentNum, rawRoughMatrix.size());
    // prepare data
    std::vector<int16_t> roughMatrix;
    roughMatrix.resize(roughCentroidNum * fragmentNum);
    std::vector<float> qdmDistance;
    qdmDistance.resize(fragmentNum * integrateCentroidNum);
    for (size_t fragmentIndex = 0; fragmentIndex < fragmentNum; ++fragmentIndex) {
        for (size_t centroid = 0; centroid < integrateCentroidNum; ++centroid) {
            // make up data
            qdmDistance[fragmentIndex * integrateCentroidNum + centroid] = fragmentIndex + centroid;
        }
    }
    std::vector<DistNode> result;
    for (size_t i = 0; i < roughCentroidNum; ++i) {
        result.emplace_back(i, 0.0f);
    }

    uint64_t t11 = Monotime::MicroSeconds();
    for (size_t blockid = 0; blockid < 8192UL / 64UL; ++blockid) {
        for (size_t fragmentIndex = 0; fragmentIndex < fragmentNum; ++fragmentIndex) {
            for (size_t centroid = 0; centroid < 64UL; ++centroid) {
                size_t realCentroid = blockid * 64 + centroid;
                const int16_t value = rawRoughMatrix[realCentroid*fragmentNum + fragmentIndex];
                roughMatrix[fragmentIndex * roughCentroidNum + realCentroid] = value;
            }
        }
    }
    fprintf(stdout, "roate-cost: %lu ", Monotime::MicroSeconds() - t11);
    // begin to compute 
    for (size_t fragmentIndex = 0; fragmentIndex <fragmentNum; ++fragmentIndex) {
        for (size_t roughCentroid = 0; roughCentroid < roughCentroidNum; ++roughCentroid) {
            size_t integrateCentroid = roughMatrix[fragmentIndex*roughCentroidNum + roughCentroid];
            result[roughCentroid].dist += qdmDistance[fragmentIndex * integrateCentroidNum + integrateCentroid];
        }
    }
    fprintf(stdout, "cost: %lu, result: ", Monotime::MicroSeconds() - t11);
    for (size_t roughCentroid = 0; roughCentroid < roughCentroidNum; ++roughCentroid) {
        if (roughCentroid % 1024 == 0) {
            fprintf(stdout, "%f, ", result[roughCentroid].dist);
        }
    }
    fprintf(stdout, "\n");
}

TEST_F(QueryDistanceMatrixTest, TestTableDistanceFragmentAsRow)
{
    size_t roughCentroidNum = 8192UL;
    size_t integrateCentroidNum = 2048UL;
    size_t fragmentNum = 64UL;
    //size_t subDimension = 8UL;
    EXPECT_EQ(roughCentroidNum * fragmentNum, rawRoughMatrix.size());
    // prepare data
    std::vector<int16_t> roughMatrix;
    for (size_t fragmentIndex = 0; fragmentIndex < fragmentNum; ++fragmentIndex) {
        for (size_t centroid = 0; centroid < roughCentroidNum; ++centroid) {
            int16_t value = rawRoughMatrix[centroid*fragmentNum + fragmentIndex];
            roughMatrix.push_back(value);
        }
    }
    std::vector<float> qdmDistance;
    qdmDistance.resize(fragmentNum * integrateCentroidNum);
    for (size_t fragmentIndex = 0; fragmentIndex < fragmentNum; ++fragmentIndex) {
        for (size_t centroid = 0; centroid < integrateCentroidNum; ++centroid) {
            // make up data
            qdmDistance[fragmentIndex * integrateCentroidNum + centroid] = fragmentIndex + centroid;
        }
    }
    std::vector<DistNode> result;
    for (size_t i = 0; i < roughCentroidNum; ++i) {
        result.emplace_back(i, 0.0f);
    }

    uint64_t t11 = Monotime::MicroSeconds();
    // begin to compute 
    for (size_t fragmentIndex = 0; fragmentIndex <fragmentNum; ++fragmentIndex) {
        for (size_t roughCentroid = 0; roughCentroid < roughCentroidNum; ++roughCentroid) {
            size_t integrateCentroid = roughMatrix[fragmentIndex*roughCentroidNum + roughCentroid];
            result[roughCentroid].dist += qdmDistance[fragmentIndex * integrateCentroidNum + integrateCentroid];
        }
    }
    fprintf(stdout, "cost: %lu, result: ", Monotime::MicroSeconds() - t11);
    for (size_t roughCentroid = 0; roughCentroid < roughCentroidNum; ++roughCentroid) {
        if (roughCentroid % 1024 == 0) {
            fprintf(stdout, "%f, ", result[roughCentroid].dist);
        }
    }
    fprintf(stdout, "\n");
}


TEST_F(QueryDistanceMatrixTest, TestNormal)
{
    return;
    string centersResource = "0,0,0.1;0,0,0.2;0,0,0.3;"
                             "0,0.1,0;0,0.2,0;0,0.3,0;"
                             "0.1,0,0;0.2,0,0;0.3,0,0;";
    CentroidResource::Pointer centerMatrixResource(new CentroidResource());
    CentroidResource::RoughMeta metaR(36, 1, {3});
    CentroidResource::IntegrateMeta meta(12, 3, 3);
    centerMatrixResource->create(metaR, meta);

    FeatureVector queryFeature{0,0,0.1,0,0.2,0,0.3,0,0};

    QueryDistanceMatrix queryDistanceMatrix(_indexMeta, centerMatrixResource.get());
    queryDistanceMatrix.init(queryFeature.data(), {1});
    return;

    //1 test queryCodeFeature
    UInt16Vector queryCodeFeature;
    queryDistanceMatrix.getQueryCodeFeature(queryCodeFeature);
    UInt16Vector left_QueryCodeFeature;
    left_QueryCodeFeature.push_back(0);
    left_QueryCodeFeature.push_back(1);
    left_QueryCodeFeature.push_back(2);

    ASSERT_EQ((size_t)3, queryCodeFeature.size()) << "code feature size";
    for (size_t i = 0 ; i < 3 ; ++i)
    {
        EXPECT_EQ(left_QueryCodeFeature[i], queryCodeFeature[i]);
    }

    const distance_t LeftDistanceMatrix[] = {
        0, 0.01, 0.04,
        0.01, 0, 0.01,
        0.04, 0.01, 0
    };
    //2 test QDM table
    distance_t distance = 0;
    for (uint16_t i = 0 ; i < 3 ; ++i) 
    {
        for (uint16_t j = 0 ; j < 3 ;++j ) 
        {
            // TODO double check
            distance = queryDistanceMatrix.getDistance(j,i);
            float offset = abs(LeftDistanceMatrix[i*3 + j] - distance);
            ASSERT_TRUE(offset < 0.00001);
        }
    }

    //3 test get row and col
    size_t centroidNum = queryDistanceMatrix.getCentroidNum();
    size_t fragmentNum = queryDistanceMatrix.getFragmentNum();
    ASSERT_EQ(size_t(meta.fragmentNum), fragmentNum);
    ASSERT_EQ(size_t(meta.centroidNum), centroidNum);
}

TEST_F(QueryDistanceMatrixTest, TestDifferentDim)
{
    return;
    string centersResource = "0,0,0.1;0,0,0.2;0,0,0.3;"
                             "0,0.1,0;0,0.2,0;0,0.3,0;"
                             "0.1,0,0;0.2,0,0;0.3,0,0;";
    CentroidResource::Pointer centerMatrixResource(
            new CentroidResource());
    CentroidResource::RoughMeta metaR(36, 1, {3});
    CentroidResource::IntegrateMeta meta(12, 3, 3);
    centerMatrixResource->create(metaR, meta);

    FeatureVector queryFeature{0,0,0.1,0,0.2,0,0.3,0,0};
    QueryDistanceMatrix queryDistanceMatrix(_indexMeta, centerMatrixResource.get());
    queryDistanceMatrix.init(queryFeature.data(), {1});

    //1 test queryCodeFeature
    UInt16Vector queryCodeFeature;
    queryDistanceMatrix.getQueryCodeFeature(queryCodeFeature);
    UInt16Vector left_QueryCodeFeature;
    left_QueryCodeFeature.push_back(0);
    left_QueryCodeFeature.push_back(1);
    left_QueryCodeFeature.push_back(2);

    ASSERT_EQ((size_t)3, queryCodeFeature.size());
    for (size_t i = 0 ; i < 3 ; ++i)
    {
        ASSERT_EQ(left_QueryCodeFeature[i], queryCodeFeature[i]);
    }

    distance_t LeftDistanceMatrix[] = {0, 0.01, 0.04,       \
                                       0.01, 0, 0.01,       \
                                       0.04, 0.01, 0};
    //2 test QDM table
    distance_t distance = 0;
    for (uint16_t i = 0 ; i < 3 ; ++i) 
    {
        for (uint16_t j = 0 ; j < 3 ;++j ) 
        {
            distance = queryDistanceMatrix.getDistance(j,i);
            float offset = abs(LeftDistanceMatrix[i*3 + j] - distance);
            ASSERT_TRUE(offset < 0.00001);
        }
    }

    //3 test get row and col
    uint16_t row = queryDistanceMatrix.getFragmentNum();
    uint16_t col = queryDistanceMatrix.getCentroidNum();
    ASSERT_EQ(meta.fragmentNum, row);
    ASSERT_EQ(meta.centroidNum, col);
}

TEST_F(QueryDistanceMatrixTest, TestDifferentMeta)
{
    return;
    string centersResource = "0,0,0;1,1,1;0,0,0;1,0,1;";
    CentroidResource::Pointer centerMatrixResource(
            new CentroidResource());
    CentroidResource::RoughMeta metaR(24, 1, {2});
    CentroidResource::IntegrateMeta meta(12, 2, 2);
    centerMatrixResource->create(metaR, meta);

    FeatureVector queryFeature{0,0,0,0,0,0};
    QueryDistanceMatrix queryDistanceMatrix(_indexMeta, centerMatrixResource.get());
    queryDistanceMatrix.init(queryFeature.data(), {1});

    //1 test queryCodeFeature
    UInt16Vector queryCodeFeature;
    queryDistanceMatrix.getQueryCodeFeature(queryCodeFeature);
    UInt16Vector left_QueryCodeFeature;
    left_QueryCodeFeature.push_back(0);
    left_QueryCodeFeature.push_back(0);

    ASSERT_EQ((size_t)2, queryCodeFeature.size());
    for (size_t i = 0 ; i < 2 ; ++i)
    {
        ASSERT_EQ(left_QueryCodeFeature[i], queryCodeFeature[i]);
    }

    distance_t LeftDistanceMatrix[] = {0, 3, 0, 2};
    //2 test QDM table
    distance_t distance = 0;
    for (uint16_t i = 0 ; i < 2; ++i) 
    {
        for (uint16_t j = 0 ; j < 2;++j ) 
        {
            distance = queryDistanceMatrix.getDistance(j,i);
            float offset = abs(LeftDistanceMatrix[i*2 + j] - distance);
            ASSERT_TRUE(offset < 0.00001);
        }
    }

    //3 test get row and col
    size_t row = queryDistanceMatrix.getFragmentNum();
    size_t col = queryDistanceMatrix.getCentroidNum();
    ASSERT_EQ((size_t)meta.fragmentNum, row);
    ASSERT_EQ((size_t)meta.centroidNum, col);
}

TEST_F(QueryDistanceMatrixTest, TestDifferentMeta2)
{
    //string centersResource = "0,0,0,1;0,0,0,2;0,0,0,3;"
    //                         "0,0,0,4;0,0,0,5;0,0,0,6;";

    //MockCentroidResource::Pointer centerMatrixResource(
    //        new MockCentroidResource());
    //CentroidResource::IntegrateMeta meta;
    //meta._fragmentNum = 3;
    //meta._centroidNum = 2;
    //meta._subDimension = 4;
    //centerMatrixResource->setIntegrateMeta(meta);
    //centerMatrixResource->setIntegrateResource(centersResource);
    //                      
    //FeatureVector queryFeature{0,0,0,0,0,0,0,0,0,0,0,0};
    //QueryDistanceMatrix queryDistanceMatrix(_indexMeta, centerMatrixResource);
    //queryDistanceMatrix.init(queryFeature);

    ////1 test queryCodeFeature
    //UInt16Vector queryCodeFeature;
    //queryDistanceMatrix.getQueryCodeFeature(queryCodeFeature);
    //UInt16Vector left_QueryCodeFeature;
    //left_QueryCodeFeature.push_back(0);
    //left_QueryCodeFeature.push_back(0);
    //left_QueryCodeFeature.push_back(0);
    //
    //ASSERT_EQ((size_t)3, queryCodeFeature.size());
    //for (size_t i = 0 ; i < 3 ; ++i)
    //{
    //    ASSERT_EQ(left_QueryCodeFeature[i], queryCodeFeature[i]);
    //}

    //distance_t LeftDistanceMatrix[] = {1, 4,
    //                                   9, 16,
    //                                   25, 36};
    ////2 test QDM table
    //distance_t distance = 0;
    //for (uint16_t i = 0 ; i < 3 ; ++i) 
    //{
    //    for (uint16_t j = 0 ; j < 2 ;++j ) 
    //    {
    //        distance = queryDistanceMatrix.getDistance(i,j);
    //        float offset = abs(LeftDistanceMatrix[i*2 + j] - distance);
    //        ASSERT_TRUE(offset < 0.00001);
    //    }
    //}

    ////3 test get row and col
    //size_t row = queryDistanceMatrix.getFragmentNum();
    //size_t col = queryDistanceMatrix.getCentroidNum();
    //ASSERT_EQ((size_t)meta._fragmentNum, row);
    //ASSERT_EQ((size_t)meta._centroidNum, col);
}

