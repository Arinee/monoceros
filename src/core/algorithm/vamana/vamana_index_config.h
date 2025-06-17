#pragma once

#include "src/core/framework/index_framework.h"
#include "src/core/algorithm/vamana/vamana_params.h"
#include "src/core/framework/index_logger.h"
#include "src/core/common/common.h"


MERCURY_NAMESPACE_BEGIN(core);

struct IndexConfig
{
    size_t dimension;
    size_t max_points;
    size_t max_reserve_degree;

    bool dynamic_index;
    bool enable_tags;
    bool pq_dist_build;
    bool concurrent_consolidate;
    bool use_opq;
    bool filtered_index;

    size_t num_pq_chunks;
    size_t num_frozen_pts;

    // Params for building index
    std::shared_ptr<IndexWriteParameters> index_write_params;
    // Params for searching index
    std::shared_ptr<IndexSearchParams> index_search_params;

  private:
    IndexConfig(size_t dimension, size_t max_points, size_t max_reserve_degree, size_t num_pq_chunks, size_t num_frozen_points, 
                bool dynamic_index, bool enable_tags, bool pq_dist_build, bool concurrent_consolidate, 
                bool use_opq, bool filtered_index, std::shared_ptr<IndexWriteParameters> index_write_params,
                std::shared_ptr<IndexSearchParams> index_search_params)
        : dimension(dimension), max_points(max_points), max_reserve_degree(max_reserve_degree), dynamic_index(dynamic_index), 
          enable_tags(enable_tags), pq_dist_build(pq_dist_build), concurrent_consolidate(concurrent_consolidate), use_opq(use_opq), filtered_index(filtered_index),
          num_pq_chunks(num_pq_chunks), num_frozen_pts(num_frozen_points), index_write_params(index_write_params), index_search_params(index_search_params)
    {
    }

    friend class IndexConfigBuilder;
};

class IndexConfigBuilder
{
  public:
    IndexConfigBuilder() = default;

    IndexConfigBuilder &with_dimension(size_t dimension)
    {
        this->_dimension = dimension;
        return *this;
    }

    IndexConfigBuilder &with_max_points(size_t max_points)
    {
        this->_max_points = max_points;
        return *this;
    }

    IndexConfigBuilder &with_max_reserve_degree(size_t max_reserve_degree)
    {
        this->_max_reserve_degree = max_reserve_degree;
        return *this;
    }

    IndexConfigBuilder &is_dynamic_index(bool dynamic_index)
    {
        this->_dynamic_index = dynamic_index;
        return *this;
    }

    IndexConfigBuilder &is_enable_tags(bool enable_tags)
    {
        this->_enable_tags = enable_tags;
        return *this;
    }

    IndexConfigBuilder &is_pq_dist_build(bool pq_dist_build)
    {
        this->_pq_dist_build = pq_dist_build;
        return *this;
    }

    IndexConfigBuilder &is_concurrent_consolidate(bool concurrent_consolidate)
    {
        this->_concurrent_consolidate = concurrent_consolidate;
        return *this;
    }

    IndexConfigBuilder &is_use_opq(bool use_opq)
    {
        this->_use_opq = use_opq;
        return *this;
    }

    IndexConfigBuilder &is_filtered(bool is_filtered)
    {
        this->_filtered_index = is_filtered;
        return *this;
    }

    IndexConfigBuilder &with_num_pq_chunks(size_t num_pq_chunks)
    {
        this->_num_pq_chunks = num_pq_chunks;
        return *this;
    }

    IndexConfigBuilder &with_num_frozen_pts(size_t num_frozen_pts)
    {
        this->_num_frozen_pts = num_frozen_pts;
        return *this;
    }

    IndexConfigBuilder &with_index_write_params(IndexWriteParameters &index_write_params)
    {
        this->_index_write_params = std::make_shared<IndexWriteParameters>(index_write_params);
        return *this;
    }

    IndexConfigBuilder &with_index_write_params(std::shared_ptr<IndexWriteParameters> index_write_params_ptr)
    {
        if (index_write_params_ptr == nullptr)
        {
            LOG_ERROR("Passed, empty build_params while creating index config");
            return *this;
        }
        this->_index_write_params = index_write_params_ptr;
        return *this;
    }

    IndexConfigBuilder &with_index_search_params(IndexSearchParams &search_params)
    {
        this->_index_search_params = std::make_shared<IndexSearchParams>(search_params);
        return *this;
    }

    IndexConfigBuilder &with_index_search_params(std::shared_ptr<IndexSearchParams> search_params_ptr)
    {
        if (search_params_ptr == nullptr)
        {
            LOG_ERROR("Passed, empty search_params while creating index config");
            return *this;
        }
        this->_index_search_params = search_params_ptr;
        return *this;
    }

    IndexConfig build()
    {
        if (_dynamic_index && _num_frozen_pts == 0)
        {
            _num_frozen_pts = 1;
        }

        if (_dynamic_index)
        {
            if (_index_search_params != nullptr && _index_search_params->initial_search_list_size == 0)
                throw std::runtime_error("Error: please pass initial_search_list_size for building dynamic index.");
        }

        // sanity check
        if (_dynamic_index && _num_frozen_pts == 0)
        {
            LOG_ERROR("_num_frozen_pts passed as 0 for dynamic_index. Setting it to 1 for safety.");
            _num_frozen_pts = 1;
        }

        return IndexConfig(_dimension, _max_points, _max_reserve_degree, _num_pq_chunks,
                           _num_frozen_pts, _dynamic_index, _enable_tags, _pq_dist_build, _concurrent_consolidate,
                           _use_opq, _filtered_index, _index_write_params, _index_search_params);
    }

    IndexConfigBuilder(const IndexConfigBuilder &) = delete;
    IndexConfigBuilder &operator=(const IndexConfigBuilder &) = delete;

  private:

    size_t _dimension;
    size_t _max_points;
    size_t _max_reserve_degree;

    bool _dynamic_index = false;
    bool _enable_tags = false;
    bool _pq_dist_build = false;
    bool _concurrent_consolidate = false;
    bool _use_opq = false;
    bool _filtered_index{defaults::HAS_LABELS};

    size_t _num_pq_chunks = 0;
    size_t _num_frozen_pts{defaults::NUM_FROZEN_POINTS_STATIC};

    std::shared_ptr<IndexWriteParameters> _index_write_params;
    std::shared_ptr<IndexSearchParams> _index_search_params;
};

MERCURY_NAMESPACE_END(core);

