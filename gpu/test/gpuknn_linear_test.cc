/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     knn_linear_test.cc
 *   \author   Hechong.xyf
 *   \date     Apr 2018
 *   \version  1.0.0
 *   \brief    Implementation of Aitheta Knn Linear Test
 */

#include "aitheta/index_framework.h"
#include "aitheta/utility/time_helper.h"
#include <gtest/gtest.h>
#include <random>

TEST(GpuKnnLinear, General)
{
    aitheta::IndexPluginBroker broker;
    broker.load("../build/lib/*.dylib");
    broker.load("../build/lib/*.so");
    broker.load("../build/lib/*.dll");
    broker.load("../build/lib/*.dylib");
    broker.load("../lib/*.so");
    broker.load("../lib/*.dll");
    ASSERT_TRUE(broker.count() > 0);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0, 1.0f);

    const uint32_t count = 10000u;
    const uint32_t dimension = 256u;

    aitheta::IndexStorage::Pointer storage =
        aitheta::IndexFactory::CreateStorage("MMapFileStorage");
    ASSERT_TRUE(!!storage);

    {
        // Prepare data in holder
        aitheta::FloatVectorHolder<dimension>::Pointer float_holder(
            new aitheta::FloatVectorHolder<dimension>);

        for (uint32_t i = 0; i < count; ++i) {
            aitheta::FloatFixedVector<dimension> val;
            for (uint32_t j = 0; j < dimension; ++j) {
                val[j] = dist(gen);
            }
            float_holder->emplace(i + 1, val);
        }

        aitheta::IndexBuilder::Pointer builder =
            aitheta::IndexFactory::CreateBuilder("KnnLinearBuilder");
        ASSERT_TRUE(!!builder);

        aitheta::IndexMeta index_meta;
        index_meta.setMeta(float_holder->type(), float_holder->dimension());
        index_meta.setMethod(
              aitheta::IndexDistance::kMethodFloatSquaredEuclidean);
        ASSERT_TRUE(index_meta.isMatched(*float_holder));

        EXPECT_EQ(0, builder->init(index_meta, aitheta::IndexParams()));
        EXPECT_EQ(0, builder->trainIndex(float_holder));
        EXPECT_EQ(0, builder->buildIndex(float_holder));
        EXPECT_EQ(0, builder->dumpIndex("./", storage));
        EXPECT_EQ(0, builder->cleanup());
    }

    {
        aitheta::IndexSearcher::Pointer searcher =
            aitheta::IndexFactory::CreateSearcher("KnnLinearSearcher");
        ASSERT_TRUE(!!searcher);

        aitheta::IndexParams params;
        params.set("knn.linear.searcher.fastload", false);
        EXPECT_EQ(0, searcher->init(params));

        EXPECT_EQ(0, searcher->loadIndex("./", storage));

        aitheta::IndexSearcher::Context::Pointer context =
            searcher->createContext(params);

        aitheta::IndexSearcher::Pointer searcher1 =
            aitheta::IndexFactory::CreateSearcher("GpuKnnLinearSearcher");
        ASSERT_TRUE(!!searcher1);

        aitheta::IndexParams params1;
        params1.set("knn.linear.searcher.fastload", false);
        params1.set("knn.linear.searcher.gpu.device.no", 0);
        EXPECT_EQ(0, searcher1->init(params1));

        EXPECT_EQ(0, searcher1->loadIndex("./", storage));

        aitheta::IndexSearcher::Context::Pointer context1 =
            searcher1->createContext(params1);


        for (size_t i = 0; i < 1; ++i) {
            // Do a query
            aitheta::FloatFixedVector<dimension> query;
            for (uint32_t j = 0; j < dimension; ++j) {
                query[j] = 0.711f;
            }

            uint64_t t11 = aitheta::Monotime::MicroSeconds();
            EXPECT_EQ(
                0, searcher->knnSearch(10, query.data(), dimension, context));
            EXPECT_EQ(
                0, searcher1->knnSearch(10, query.data(), dimension, context1));

            int size1 = context->result().size();
            int size2 = context1->result().size();
            EXPECT_EQ(size1, 10);
            EXPECT_EQ(size2, 10);
            for (int i = 0; i < size1; i++) {
                EXPECT_EQ(context->result()[i].index, context1->result()[i].index);
                EXPECT_EQ(context->result()[i].key, context1->result()[i].key);
            }
            /*
            for (auto &item : context->result()) {
                std::cout << '[' << item.index << "] " << item.key << ": "
                          << item.score << std::endl;
            }
            for (auto &item : context1->result()) {
                std::cout << '[' << item.index << "] " << item.key << ": "
                          << item.score << std::endl;
            }
            */
        }

        EXPECT_EQ(0, searcher->unloadIndex());
        EXPECT_EQ(0, searcher->cleanup());
        EXPECT_EQ(0, searcher1->unloadIndex());
        EXPECT_EQ(0, searcher1->cleanup());
    }

    {
        // Prepare data in holder
        aitheta::FloatVectorHolder<dimension>::Pointer float_holder(
            new aitheta::FloatVectorHolder<dimension>);

        for (uint32_t i = 0; i < count; ++i) {
            aitheta::FloatFixedVector<dimension> val;
            for (uint32_t j = 0; j < dimension; ++j) {
                val[j] = dist(gen);
            }
            float_holder->emplace(i + 1, val);
        }

        aitheta::IndexBuilder::Pointer builder =
            aitheta::IndexFactory::CreateBuilder("KnnLinearBuilder");
        ASSERT_TRUE(!!builder);

        aitheta::IndexMeta index_meta;
        index_meta.setMeta(float_holder->type(), float_holder->dimension());
        index_meta.setMethod(
              aitheta::IndexDistance::kMethodFloatInnerProduct);
        ASSERT_TRUE(index_meta.isMatched(*float_holder));

        EXPECT_EQ(0, builder->init(index_meta, aitheta::IndexParams()));
        EXPECT_EQ(0, builder->trainIndex(float_holder));
        EXPECT_EQ(0, builder->buildIndex(float_holder));
        EXPECT_EQ(0, builder->dumpIndex("./", storage));
        EXPECT_EQ(0, builder->cleanup());
    }

    {
        aitheta::IndexSearcher::Pointer searcher =
            aitheta::IndexFactory::CreateSearcher("KnnLinearSearcher");
        ASSERT_TRUE(!!searcher);

        aitheta::IndexParams params;
        params.set("knn.linear.searcher.fastload", false);
        EXPECT_EQ(0, searcher->init(params));

        EXPECT_EQ(0, searcher->loadIndex("./", storage));

        aitheta::IndexSearcher::Context::Pointer context =
            searcher->createContext(params);

        aitheta::IndexSearcher::Pointer searcher1 =
            aitheta::IndexFactory::CreateSearcher("GpuKnnLinearSearcher");
        ASSERT_TRUE(!!searcher1);

        aitheta::IndexParams params1;
        params1.set("knn.linear.searcher.fastload", false);
        params1.set("knn.linear.searcher.gpu.device.no", 0);
        EXPECT_EQ(0, searcher1->init(params1));

        EXPECT_EQ(0, searcher1->loadIndex("./", storage));

        aitheta::IndexSearcher::Context::Pointer context1 =
            searcher1->createContext(params1);


        for (size_t i = 0; i < 1; ++i) {
            // Do a query
            aitheta::FloatFixedVector<dimension> query;
            for (uint32_t j = 0; j < dimension; ++j) {
                query[j] = 0.711f;
            }

            uint64_t t11 = aitheta::Monotime::MicroSeconds();
            EXPECT_EQ(
                0, searcher->knnSearch(10, query.data(), dimension, context));
            EXPECT_EQ(
                0, searcher1->knnSearch(10, query.data(), dimension, context1));

            int size1 = context->result().size();
            int size2 = context1->result().size();
            EXPECT_EQ(size1, 10);
            EXPECT_EQ(size2, 10);
            for (int i = 0; i < size1; i++) {
                EXPECT_EQ(context->result()[i].index, context1->result()[i].index);
                EXPECT_EQ(context->result()[i].key, context1->result()[i].key);
            }
            /*
            for (auto &item : context->result()) {
                std::cout << '[' << item.index << "] " << item.key << ": "
                          << item.score << std::endl;
            }
            for (auto &item : context1->result()) {
                std::cout << '[' << item.index << "] " << item.key << ": "
                          << item.score << std::endl;
            }
            */
        }

        EXPECT_EQ(0, searcher->unloadIndex());
        EXPECT_EQ(0, searcher->cleanup());
        EXPECT_EQ(0, searcher1->unloadIndex());
        EXPECT_EQ(0, searcher1->cleanup());
    }
}
