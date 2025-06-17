/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     in_mem_graph_store.cc
 *   \author   shiyang
 *   \date     Nov 2023
 *   \version  1.0.0
 *   \brief    interface and impl of in memory data store
 */

#include "in_mem_graph_store.h"

MERCURY_NAMESPACE_BEGIN(core);

InMemGraphStore::InMemGraphStore(const size_t total_pts, const size_t reserve_graph_degree) 
    : _capacity(total_pts), _reserve_graph_degree(reserve_graph_degree)
{
    this->resize_graph(total_pts);
    for (size_t i = 0; i < total_pts; i++)
    {
        _graph[i].reserve(reserve_graph_degree);
    }
}

InMemGraphStore::~InMemGraphStore()
{
    clear_graph();
    if (_graph_array != nullptr) {
        delete[] _graph_array;
    }
}

size_t InMemGraphStore::resize_graph(const size_t new_size)
{
    _graph.resize(new_size);
    set_total_points(new_size);
    return _graph.size();
}

void InMemGraphStore::clear_graph()
{
    _graph.clear();
}

std::tuple<uint32_t, uint32_t, size_t> InMemGraphStore::load_impl(const std::string &filename, size_t expected_num_points)
{
    size_t expected_file_size;
    size_t file_frozen_pts;
    uint32_t start;
    size_t file_offset = 0; // will need this for single file format support

    std::ifstream in;
    in.exceptions(std::ios::badbit | std::ios::failbit);
    in.open(filename, std::ios::binary);
    in.seekg(file_offset, in.beg);
    in.read((char *)&expected_file_size, sizeof(size_t));
    in.read((char *)&_max_observed_degree, sizeof(uint32_t));
    in.read((char *)&start, sizeof(uint32_t));
    in.read((char *)&file_frozen_pts, sizeof(size_t));
    size_t vamana_metadata_size = sizeof(size_t) + sizeof(uint32_t) + sizeof(uint32_t) + sizeof(size_t);

    LOG_INFO("From graph header, expected_file_size: %lu, _max_observed_degree: %d, _start: %d, file_frozen_pts: %lu", 
                expected_file_size, _max_observed_degree, start, file_frozen_pts);

    LOG_INFO("Loading vamana graph %s...", filename.c_str());

    // If user provides more points than max_points
    // resize the _graph to the larger size.
    if (get_total_points() < expected_num_points)
    {
        LOG_INFO("resizing graph to %lu", expected_num_points);
        this->resize_graph(expected_num_points);
    }

    size_t bytes_read = vamana_metadata_size;
    size_t cc = 0;
    uint32_t nodes_read = 0;
    while (bytes_read != expected_file_size)
    {
        uint32_t k;
        in.read((char *)&k, sizeof(uint32_t));

        if (k == 0)
        {
            LOG_ERROR("ERROR: Point found with no out-neighbours, point# %d", nodes_read)
        }

        cc += k;
        ++nodes_read;
        std::vector<uint32_t> tmp(k);
        tmp.reserve(k);
        in.read((char *)tmp.data(), k * sizeof(uint32_t));
        _graph[nodes_read - 1].swap(tmp);
        bytes_read += sizeof(uint32_t) * ((size_t)k + 1);
        if (nodes_read % 10000000 == 0)
            LOG_INFO(".");
        if (k > _max_range_of_graph)
        {
            _max_range_of_graph = k;
        }
    }

    LOG_INFO("done. Index has %d nodes and %lu out-edges, _start is set to %d", nodes_read, cc, start);
    return std::make_tuple(nodes_read, start, file_frozen_pts);
}

std::tuple<uint32_t, uint32_t, size_t> InMemGraphStore::load(const std::string &index_path_prefix,
                                                            const size_t num_points)
{
    return load_impl(index_path_prefix, num_points);
}

int InMemGraphStore::save_graph(const std::string &index_path_prefix, const size_t num_points,
                            const size_t num_frozen_points, const uint32_t start)
{
    std::ofstream out;
    open_file_to_write(out, index_path_prefix);

    size_t file_offset = 0;
    out.seekp(file_offset, out.beg);
    size_t index_size = 24;
    uint32_t max_degree = 0;
    out.write((char *)&index_size, sizeof(uint64_t));
    out.write((char *)&_max_observed_degree, sizeof(uint32_t));
    uint32_t ep_u32 = start;
    out.write((char *)&ep_u32, sizeof(uint32_t));
    out.write((char *)&num_frozen_points, sizeof(size_t));

    // Note: num_points = _nd + _num_frozen_points
    LOG_INFO("num of points: %lu", num_points);
    for (uint32_t i = 0; i < num_points; i++)
    {
        uint32_t GK = (uint32_t)_graph[i].size();
        out.write((char *)&GK, sizeof(uint32_t));
        out.write((char *)_graph[i].data(), GK * sizeof(uint32_t));
        max_degree = _graph[i].size() > max_degree ? (uint32_t)_graph[i].size() : max_degree;
        index_size += (size_t)(sizeof(uint32_t) * (GK + 1));
    }
    LOG_INFO("index_size: %lu", index_size);
    out.seekp(file_offset, out.beg);
    out.write((char *)&index_size, sizeof(uint64_t));
    out.write((char *)&max_degree, sizeof(uint32_t));
    out.close();
    return (int)index_size;
}

int InMemGraphStore::store(const std::string &index_path_prefix, const size_t num_points,
                        const size_t num_frozen_points, const uint32_t start)
{
    return save_graph(index_path_prefix, num_points, num_frozen_points, start);
}

void InMemGraphStore::dump(IndexPackage &index_package, const size_t num_points,
                           const size_t num_frozen_points, const uint32_t start)
{
    uint32_t max_degree = 0;
    uint32_t ep_u32 = start;
    size_t graph_size = 0;
    uint32_t graph_pts = 0;
    uint32_t node_degrees[num_points];
    for (uint32_t i = 0; i < num_points; i++)
    {
        uint32_t degree = (uint32_t)_graph[i].size();
        node_degrees[i] = degree;
        max_degree = degree > max_degree ? degree : max_degree;
        graph_size += (size_t)(sizeof(uint32_t) * degree);
        graph_pts += degree;
    }
    
    _graph_array = new uint32_t[graph_pts];
    uint32_t graph_index = 0;
    for (uint32_t i = 0; i < num_points; i++)
    {
        for (uint32_t j = 0; j < node_degrees[i]; j++) {
            _graph_array[graph_index++] = _graph[i][j];
        }
    }
    LOG_INFO("graph_pts: %d; graph_index: %d; graph_size: %lu", graph_pts, graph_index, graph_size);
    index_package.emplace("graph_store_max_degree", (const void *)&max_degree, sizeof(int));
    index_package.emplace("graph_store_start", (const void *)&ep_u32, sizeof(int));
    index_package.emplace("graph_num_frozen_points", (const void *)&num_frozen_points, sizeof(size_t));
    index_package.emplace("graph_node_degrees", (const void *)&node_degrees, num_points * sizeof(uint32_t));
    index_package.emplace("graph_data_array", static_cast<const void*>(_graph_array), graph_size);
}

const std::vector<uint32_t> InMemGraphStore::get_neighbours(const uint32_t i) const
{
    return _graph.at(i);
}

void InMemGraphStore::add_neighbour(const uint32_t i, uint32_t neighbour_id)
{
    _graph[i].emplace_back(neighbour_id);
    if (_max_observed_degree < _graph[i].size())
    {
        _max_observed_degree = (uint32_t)(_graph[i].size());
    }
}

void InMemGraphStore::clear_neighbours(const uint32_t i)
{
    _graph[i].clear();
};

void InMemGraphStore::swap_neighbours(const uint32_t a, uint32_t b)
{
    _graph[a].swap(_graph[b]);
};

void InMemGraphStore::set_neighbours(const uint32_t i, std::vector<uint32_t> &neighbours)
{
    _graph[i].assign(neighbours.begin(), neighbours.end());
    if (_max_observed_degree < neighbours.size())
    {
        _max_observed_degree = (uint32_t)(neighbours.size());
    }
}

size_t InMemGraphStore::get_total_points()
{
    return _capacity;
}

size_t InMemGraphStore::get_max_range_of_graph()
{
    return _max_range_of_graph;
}

uint32_t InMemGraphStore::get_max_observed_degree()
{
    return _max_observed_degree;
}

void InMemGraphStore::set_total_points(size_t new_capacity)
{
    _capacity = new_capacity;
}

MERCURY_NAMESPACE_END(core);
