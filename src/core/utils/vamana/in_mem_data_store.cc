/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     in_mem_data_store.cc
 *   \author   shiyang
 *   \date     Nov 2023
 *   \version  1.0.0
 *   \brief    interface and impl of in memory data store
 */

#include "in_mem_data_store.h"

MERCURY_NAMESPACE_BEGIN(core);

InMemDataStore::InMemDataStore(const uint32_t capacity, const size_t dim, const uint16_t data_size) 
    : _capacity(capacity), _dim(dim), _data_size(data_size)
{
    _aligned_dim = ROUND_UP(dim, DEFAULT_ALIGNMENT_FACTOR);
    alloc_aligned(((void **)&_data), this->_capacity * _aligned_dim * _data_size, 8 * _data_size);
    std::memset((char *)_data, 0, this->_capacity * _aligned_dim * _data_size);
}

InMemDataStore::~InMemDataStore()
{
    if (_data != nullptr)
    {
        aligned_free(this->_data);
    }
}

void InMemDataStore::setMethod(IndexDistance::Methods method)
{
    _method = method;
    this->updateMeasure();
}

uint32_t InMemDataStore::capacity() const
{
    return _capacity;
}

size_t InMemDataStore::get_dims() const
{
    return _dim;
}

uint32_t InMemDataStore::load(const std::string &filename) 
{
    if (!file_exists(filename))
    {
        LOG_ERROR("ERROR: data file %s does not exist.", filename.c_str());
        aligned_free(_data);
        std::stringstream stream;
        stream << "ERROR: data file " << filename << " does not exist." << std::endl;
        throw new std::runtime_error(stream.str());
    }
    size_t file_dim, file_num_points;
    get_bin_metadata(filename, file_num_points, file_dim);

    if (file_dim != this->_dim)
    {
        std::stringstream stream;
        stream << "ERROR: Driver requests loading " << this->_dim << " dimension,"
            << "but file has " << file_dim << " dimension." << std::endl;
        aligned_free(_data);
        throw new std::runtime_error(stream.str());
    }

    if (file_num_points > this->_capacity)
    {
        this->resize((uint32_t)file_num_points);
    }

    copy_aligned_data_from_file(_data_size, filename.c_str(), _data, file_num_points, file_dim, _aligned_dim);

    return (uint32_t)file_num_points;
}

size_t InMemDataStore::save(const std::string &filename, const uint32_t num_points)
{
    return save_data_in_base_dimensions(_data_size, filename, _data, num_points, this->get_dims(), this->get_aligned_dim(), 0U);
}

void InMemDataStore::dump(IndexPackage &index_package, const uint32_t num_points)
{
    int npts_i32 = num_points, ndims_i32 = (int)(this->get_dims());
    index_package.emplace("data_store_num_pts", (const void *)&npts_i32, sizeof(int));
    index_package.emplace("data_store_num_dim", (const void *)&ndims_i32, sizeof(int));
    index_package.emplace("data_store_data", (const void *)_data, npts_i32 * ndims_i32 * _data_size);
}

uint32_t InMemDataStore::resize(const uint32_t new_num_points)
{
    if (new_num_points > _capacity)
    {
        return expand(new_num_points);
    }
    else if (new_num_points < _capacity)
    {
        return shrink(new_num_points);
    }
    else
    {
        return _capacity;
    }
}

uint32_t InMemDataStore::expand(const uint32_t new_size)
{
    if (new_size == this->_capacity)
    {
        return this->capacity();
    }
    else if (new_size < this->capacity())
    {
        LOG_ERROR("Cannot 'expand' datastore when new capacity (%d) < existing capacity(%d)", new_size, this->capacity());
        std::stringstream ss;
        ss << "Cannot 'expand' datastore when new capacity (" << new_size << ") < existing capacity("
        << this->capacity() << ")" << std::endl;
        throw new std::runtime_error(ss.str());
    }
    void *new_data;
    alloc_aligned((void **)&new_data, new_size * _aligned_dim * _data_size, 8 * _data_size);
    memcpy(new_data, _data, this->capacity() * _aligned_dim * _data_size);
    aligned_free(_data);
    _data = new_data;
    this->_capacity = new_size;
    return this->_capacity;
}

uint32_t InMemDataStore::shrink(const uint32_t new_size)
{
    if (new_size == this->capacity())
    {
        return this->capacity();
    }
    else if (new_size > this->capacity())
    {
        LOG_ERROR("Cannot 'shrink' datastore when new capacity (%d) < existing capacity(%d)", new_size, this->capacity());
        std::stringstream ss;
        ss << "Cannot 'shrink' datastore when new capacity (" << new_size << ") > existing capacity("
        << this->capacity() << ")" << std::endl;
        throw new std::runtime_error(ss.str());
    }
    void *new_data;
    alloc_aligned((void **)&new_data, new_size * _aligned_dim * _data_size, 8 * _data_size);
    memcpy(new_data, _data, new_size * _aligned_dim * _data_size);
    aligned_free(_data);
    _data = new_data;
    this->_capacity = new_size;
    return this->_capacity;
}

void InMemDataStore::populate_data(const std::string &filename, const size_t offset)
{
    size_t npts, ndim;
    copy_aligned_data_from_file(_data_size, filename.c_str(), _data, npts, ndim, _aligned_dim, offset);

    if ((uint32_t)npts > this->capacity())
    {
        std::stringstream ss;
        ss << "Number of points in the file: " << filename
        << " is greater than the capacity of data store: " << this->capacity()
        << ". Must invoke resize before calling populate_data()" << std::endl;
        throw new std::runtime_error(ss.str());
    }

    if ((uint32_t)ndim != this->get_dims())
    {
        std::stringstream ss;
        ss << "Number of dimensions of a point in the file: " << filename
        << " is not equal to dimensions of data store: " << this->capacity() << "." << std::endl;
        throw new std::runtime_error(ss.str());
    }
}

void InMemDataStore::get_vector(const uint32_t i, void *dest) const
{
    memcpy(dest, (char *)_data + i * _aligned_dim * _data_size, this->_dim * _data_size);
}

void InMemDataStore::set_vector(const uint32_t loc, const void *const vector)
{
    size_t offset_in_data = loc * _aligned_dim;
    memset((char *)_data + offset_in_data * _data_size, 0, _aligned_dim * _data_size);
    memcpy((char *)_data + offset_in_data * _data_size, vector, this->_dim * _data_size);
}

void InMemDataStore::prefetch_vector(const uint32_t loc)
{
    prefetch_vector_impl((const char *)_data + _aligned_dim * (size_t)loc, _data_size * _aligned_dim);
}

void InMemDataStore::copy_vectors(const uint32_t from_loc, const uint32_t to_loc,
                                        const uint32_t num_points)
{
    assert(from_loc < this->_capacity);
    assert(to_loc < this->_capacity);
    assert(num_points < this->_capacity);
    memmove((char *)_data + _aligned_dim * to_loc * _data_size, (char *)_data + _aligned_dim * from_loc * _data_size, num_points * _aligned_dim * _data_size);
}

void InMemDataStore::move_vectors(const uint32_t old_location_start, const uint32_t new_location_start,
                                        const uint32_t num_locations)
{
    if (num_locations == 0 || old_location_start == new_location_start)
    {
        return;
    }

    /*    // Update pointers to the moved nodes. Note: the computation is correct
    even
        // when new_location_start < old_location_start given the C++ uint32_t
        // integer arithmetic rules.
        const uint32_t location_delta = new_location_start - old_location_start;
    */
    // The [start, end) interval which will contain obsolete points to be
    // cleared.
    uint32_t mem_clear_loc_start = old_location_start;
    uint32_t mem_clear_loc_end_limit = old_location_start + num_locations;

    if (new_location_start < old_location_start)
    {
        // If ranges are overlapping, make sure not to clear the newly copied
        // data.
        if (mem_clear_loc_start < new_location_start + num_locations)
        {
            // Clear only after the end of the new range.
            mem_clear_loc_start = new_location_start + num_locations;
        }
    }
    else
    {
        // If ranges are overlapping, make sure not to clear the newly copied
        // data.
        if (mem_clear_loc_end_limit > new_location_start)
        {
            // Clear only up to the beginning of the new range.
            mem_clear_loc_end_limit = new_location_start;
        }
    }

    // Use memmove to handle overlapping ranges.
    copy_vectors(old_location_start, new_location_start, num_locations);
    memset((char *)_data + _aligned_dim * mem_clear_loc_start * _data_size, 0,
        _data_size * _aligned_dim * (mem_clear_loc_end_limit - mem_clear_loc_start));
}

uint32_t InMemDataStore::calculate_medoid() const
{
    // allocate and init centroid
    float *center = new float[_aligned_dim];
    for (size_t j = 0; j < _aligned_dim; j++)
        center[j] = 0;

    for (size_t i = 0; i < this->capacity(); i++)
        for (size_t j = 0; j < _aligned_dim; j++)
            center[j] += ((float *)_data)[i * _aligned_dim + j];

    for (size_t j = 0; j < _aligned_dim; j++)
        center[j] /= (float)this->capacity();

    // compute all to one distance
    float *distances = new float[this->capacity()];

    for (int64_t i = 0; i < (int64_t)this->capacity(); i++)
    {
        // extract point and distance reference
        float &dist = distances[i];
        const float *cur_vec = (float *)_data + (i * (size_t)_aligned_dim);
        dist = 0;
        float diff = 0;
        for (size_t j = 0; j < _aligned_dim; j++)
        {
            diff = (center[j] - (float)cur_vec[j]) * (center[j] - (float)cur_vec[j]);
            dist += diff;
        }
    }
    // find imin
    uint32_t min_idx = 0;
    float min_dist = distances[0];
    for (uint32_t i = 1; i < this->capacity(); i++)
    {
        if (distances[i] < min_dist)
        {
            min_idx = i;
            min_dist = distances[i];
        }
    }

    delete[] distances;
    delete[] center;
    return min_idx;
}

uint32_t InMemDataStore::calculate_medoid_half() const
{
    // allocate and init centroid
    float *center = new float[_aligned_dim];
    for (size_t j = 0; j < _aligned_dim; j++)
        center[j] = 0;

    for (size_t i = 0; i < this->capacity(); i++) {
        for (size_t j = 0; j < _aligned_dim; j++) {
            center[j] += ((half_float::half *)_data)[i * _aligned_dim + j];
        }
    }

    for (size_t j = 0; j < _aligned_dim; j++)
        center[j] /= (float)this->capacity();

    // compute all to one distance
    float *distances = new float[this->capacity()];

    for (int64_t i = 0; i < (int64_t)this->capacity(); i++)
    {
        // extract point and distance reference
        float &dist = distances[i];
        const half_float::half *cur_vec = (half_float::half *)_data + (i * (size_t)_aligned_dim);
        dist = 0;
        float diff = 0;
        for (size_t j = 0; j < _aligned_dim; j++)
        {
            diff = (center[j] - (half_float::half)cur_vec[j]) * (center[j] - (half_float::half)cur_vec[j]);
            dist += diff;
        }
    }
    // find imin
    uint32_t min_idx = 0;
    float min_dist = distances[0];
    for (uint32_t i = 1; i < this->capacity(); i++)
    {
        if (distances[i] < min_dist)
        {
            min_idx = i;
            min_dist = distances[i];
        }
    }

    delete[] distances;
    delete[] center;
    return min_idx;
}

float InMemDataStore::get_distance(const void *query, const uint32_t loc) const
{
    return _measure(query, (char *)_data + _aligned_dim * loc * _data_size, _aligned_dim * _data_size);
}

float InMemDataStore::get_distance(const uint32_t loc1, const uint32_t loc2) const
{
    return _measure((char *)_data + loc1 * _aligned_dim * _data_size, (char *)_data + loc2 * _aligned_dim * _data_size, _aligned_dim * _data_size);
}

void InMemDataStore::get_distance(const void *query, const uint32_t *locations, const uint32_t location_count, float *distances) const
{
    for (uint32_t i = 0; i < location_count; i++)
    {
        distances[i] = _measure(query, (char *)_data + locations[i] * _aligned_dim * _data_size, _aligned_dim * _data_size);
    }
}

size_t InMemDataStore::get_aligned_dim() const
{
    return _aligned_dim;
}

size_t InMemDataStore::get_alignment_factor() const
{
    return DEFAULT_ALIGNMENT_FACTOR;
}

IndexDistance::Methods InMemDataStore::get_measure_method() const
{
    return this->_method;
}

void InMemDataStore::preprocess_query(const void *query_vec, const size_t query_dim, void *scratch_query)
{
    std::memcpy(scratch_query, query_vec, query_dim * _data_size);
}

void InMemDataStore::updateMeasure(void)
{
    _measure = IndexDistance::EmbodyMeasure(_method);
}

MERCURY_NAMESPACE_END(core);
