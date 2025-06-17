/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     in_mem_graph_store.h
 *   \author   shiyang
 *   \date     Nov 2023
 *   \version  1.0.0
 *   \brief    interface and impl of in memory data store
 */

#pragma once

#include <stdint.h>
#include <assert.h>
#include <string>
#include <memory>
#include <memory.h>
#include <pthread.h>
#include "define.h"
#include "utils.h"
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include "src/core/framework/index_logger.h"
#include "src/core/framework/index_package.h"
#include "src/core/common/common.h"

MERCURY_NAMESPACE_BEGIN(core);
class InMemGraphStore
{
public:
    typedef std::shared_ptr<InMemGraphStore> Pointer;

    InMemGraphStore(const size_t total_pts, const size_t reserve_graph_degree);

    ~InMemGraphStore();

    size_t resize_graph(const size_t new_size);

    void clear_graph();

    std::tuple<uint32_t, uint32_t, size_t> load_impl(const std::string &filename, size_t expected_num_points);

    std::tuple<uint32_t, uint32_t, size_t> load(const std::string &index_path_prefix, const size_t num_points);

    int save_graph(const std::string &index_path_prefix, const size_t num_points,
                                const size_t num_frozen_points, const uint32_t start);

    int store(const std::string &index_path_prefix, const size_t num_points,
                           const size_t num_frozen_points, const uint32_t start);

    void dump(IndexPackage &index_package, const size_t num_points,
                           const size_t num_frozen_points, const uint32_t start);

    const std::vector<uint32_t> get_neighbours(const uint32_t i) const;

    void add_neighbour(const uint32_t i, uint32_t neighbour_id);

    void clear_neighbours(const uint32_t i);

    void swap_neighbours(const uint32_t a, uint32_t b);

    void set_neighbours(const uint32_t i, std::vector<uint32_t> &neighbours);

    size_t get_total_points();

    size_t get_max_range_of_graph();

    uint32_t get_max_observed_degree();

private:
    size_t _capacity;
    size_t _reserve_graph_degree;
    size_t _max_range_of_graph = 0;
    uint32_t _max_observed_degree = 0;
    uint32_t * _graph_array = nullptr;

    void set_total_points(size_t new_capacity);

    std::vector<std::vector<uint32_t>> _graph;
    
};

MERCURY_NAMESPACE_END(core);
