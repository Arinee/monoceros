/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     reservoir_sampler.h
 *   \author   Hechong.xyf
 *   \date     Jun 2018
 *   \version  1.0.0
 *   \brief    Interface of Reservoir Sampler
 */

#ifndef __MERCURY_UTILITY_RESERVOIR_SAMPLER_H__
#define __MERCURY_UTILITY_RESERVOIR_SAMPLER_H__

#include <random>
#include <vector>

namespace mercury {

/*! Reservoir Sampler
 */
template <typename T, typename Allocator = std::allocator<T>>
class ReservoirSampler
{
public:
    //! Constructor
    ReservoirSampler(size_t cnt)
        : _samples(cnt), _total(0), _mt(std::random_device()()), _pool()
    {
        _pool.reserve(_samples);
    }

    //! Constructor
    ReservoirSampler(const ReservoirSampler &rhs)
        : _samples(rhs._samples), _total(rhs._total),
          _mt(std::random_device()()), _pool(rhs._pool)
    {
    }

    //! Constructor
    ReservoirSampler(ReservoirSampler &&rhs)
        : _samples(rhs._samples), _total(rhs._total),
          _mt(std::random_device()()), _pool(std::move(rhs._pool))
    {
    }

    //! Destructor
    ~ReservoirSampler(void) {}

    //! Assignment
    ReservoirSampler &operator=(const ReservoirSampler &rhs)
    {
        _samples = rhs._samples;
        _total = rhs._total;
        _pool = rhs._pool;
        return *this;
    }

    //! Assignment
    ReservoirSampler &operator=(ReservoirSampler &&rhs)
    {
        _samples = rhs._samples;
        _total = rhs._total;
        _pool = std::move(rhs._pool);
        return *this;
    }

    //! Retrieve pool of reservoir
    std::vector<T, Allocator> &pool(void)
    {
        return _pool;
    }

    //! Retrieve pool of reservoir
    const std::vector<T, Allocator> &pool(void) const
    {
        return _pool;
    }

    //! Retrieve count of samples
    size_t samples(void) const
    {
        return _samples;
    }

    //! Retrieve total count of filling
    size_t total(void) const
    {
        return _total;
    }

    //! Reset the reservoir
    void reset(void)
    {
        _total = 0;
        _pool.clear();
        _pool.reserve(_samples);
    }

    //! Fill the reservoir
    void fill(const T &item)
    {
        if (_samples > 0) {
            if (_pool.size() >= _samples) {
                std::uniform_int_distribution<size_t> dt(0, _total);
                size_t i = dt(_mt);

                if (i < _samples) {
                    _pool[i] = item;
                }
            } else {
                _pool.push_back(item);
            }
        }
        ++_total;
    }

    //! Fill the reservoir
    void fill(T &&item)
    {
        if (_samples > 0) {
            if (_pool.size() >= _samples) {
                std::uniform_int_distribution<size_t> dt(0, _total);
                size_t i = dt(_mt);

                if (i < _samples) {
                    _pool[i] = std::move(item);
                }
            } else {
                _pool.push_back(std::move(item));
            }
        }
        ++_total;
    }

private:
    //! Disable them
    ReservoirSampler(void) = delete;

    //! Members
    size_t _samples;
    size_t _total;
    std::mt19937 _mt;
    std::vector<T, Allocator> _pool;
};

} // namespace mercury

#endif // __MERCURY_UTILITY_RESERVOIR_SAMPLER_H__
