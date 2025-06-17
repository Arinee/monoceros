/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     cluster_centroid_test.cc
 *   \author   Hechong.xyf
 *   \date     Mar 2018
 *   \version  1.0.0
 *   \brief    Implementation of Cluster Centroid Test
 */

#include "cluster/cluster_centroid.h"
#include "gtest/gtest.h"
#include <random>

using namespace std;
using namespace mercury;

TEST(ClusterCentroid, General)
{
    ClusterCentroid centroid;
    std::vector<ClusterCentroid> centroid_list;

    EXPECT_EQ(0.0, centroid.score());
    EXPECT_EQ(0u, centroid.follows());

    float fvecs[12];
    for (int i = 0; i < 12; ++i) {
        fvecs[i] = (float)(i + 1);
    }
    centroid.set(fvecs, sizeof(fvecs), 800.0, 200);

    EXPECT_EQ(800.0, centroid.score());
    EXPECT_EQ(200u, centroid.follows());
    EXPECT_EQ(sizeof(fvecs), centroid.size());
    EXPECT_NE((void *)fvecs, centroid.feature());

    centroid_list.push_back(centroid);
    EXPECT_EQ(800.0, centroid.score());
    EXPECT_EQ(200u, centroid.follows());

    centroid.set(centroid.feature(), centroid.size(), 1000.0, 250);
    EXPECT_EQ(1000.0, centroid.score());
    EXPECT_EQ(250u, centroid.follows());
}
