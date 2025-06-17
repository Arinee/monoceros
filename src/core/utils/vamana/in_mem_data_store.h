/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     in_mem_data_store.h
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
#include "src/core/common/common.h"
#include "src/core/framework/index_distance.h"
#include "src/core/framework/index_package.h"

MERCURY_NAMESPACE_BEGIN(core);
class InMemDataStore
{
public:
    typedef std::shared_ptr<InMemDataStore> Pointer;

public:
    InMemDataStore(const uint32_t capacity, const size_t dim, const uint16_t data_size);

    ~InMemDataStore();

    void setMethod(IndexDistance::Methods method);

    uint32_t capacity() const;

    size_t get_dims() const;

    uint32_t load(const std::string &filename);

    size_t save(const std::string &filename, const uint32_t num_points);

    void dump(IndexPackage &index_package, const uint32_t num_points);

    uint32_t resize(const uint32_t new_num_points);

    uint32_t expand(const uint32_t new_size);

    uint32_t shrink(const uint32_t new_size);

    void populate_data(const std::string &filename, const size_t offset);

    void get_vector(const uint32_t i, void *dest) const;

    void set_vector(const uint32_t loc, const void *const vector);

    void prefetch_vector(const uint32_t loc);

    void copy_vectors(const uint32_t from_loc, const uint32_t to_loc, const uint32_t num_points);

    void move_vectors(const uint32_t old_location_start, const uint32_t new_location_start, const uint32_t num_locations);

    uint32_t calculate_medoid() const;

    uint32_t calculate_medoid_half() const;

    float get_distance(const void *query, const uint32_t loc) const;

    float get_distance(const uint32_t loc1, const uint32_t loc2) const;

    void get_distance(const void *query, const uint32_t *locations, const uint32_t location_count, float *distances) const;

    size_t get_aligned_dim() const;

    size_t get_alignment_factor() const;

    IndexDistance::Methods get_measure_method() const;

    void preprocess_query(const void *query_vec, const size_t query_dim, void *scratch_query);

private:
    void updateMeasure(void);
    
    uint32_t _capacity;

    size_t _dim;

    uint16_t _data_size;

    void *_data;

    size_t _aligned_dim;

    IndexDistance::Measure _measure;

    IndexDistance::Methods _method;
};

MERCURY_NAMESPACE_END(core);
