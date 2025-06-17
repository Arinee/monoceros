/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     buffer_sampler.h
 *   \author   Hechong.xyf
 *   \date     Jul 2018
 *   \version  1.0.0
 *   \brief    Interface of Buffer Sampler
 */

#ifndef __MERCURY_UTILITY_BUFFER_SAMPLER_H__
#define __MERCURY_UTILITY_BUFFER_SAMPLER_H__

#include <cstring>
#include <random>
#include <string>

namespace mercury {

/*! Buffer Sampler
 */
class BufferSampler
{
public:
    //! Constructor
    BufferSampler(size_t cnt, size_t ut)
        : _samples(cnt), _unit(ut), _total(0), _mt(std::random_device()()),
          _buffer()
    {
        _buffer.reserve(_samples * _unit);
    }

    //! Constructor
    BufferSampler(const BufferSampler &rhs)
        : _samples(rhs._samples), _unit(rhs._unit), _total(rhs._total),
          _mt(std::random_device()()), _buffer(rhs._buffer)
    {
    }

    //! Constructor
    BufferSampler(BufferSampler &&rhs)
        : _samples(rhs._samples), _unit(rhs._unit), _total(rhs._total),
          _mt(std::random_device()()), _buffer(std::move(rhs._buffer))
    {
    }

    //! Destructor
    ~BufferSampler(void) {}

    //! Assignment
    BufferSampler &operator=(const BufferSampler &rhs)
    {
        _samples = rhs._samples;
        _unit = rhs._unit;
        _total = rhs._total;
        _buffer = rhs._buffer;
        return *this;
    }

    //! Assignment
    BufferSampler &operator=(BufferSampler &&rhs)
    {
        _samples = rhs._samples;
        _unit = rhs._unit;
        _total = rhs._total;
        _buffer = std::move(rhs._buffer);
        return *this;
    }

    //! Retrieve buffer
    std::string &buffer(void)
    {
        return _buffer;
    }

    //! Retrieve buffer
    const std::string &buffer(void) const
    {
        return _buffer;
    }

    //! Retrieve count of samples
    size_t samples(void) const
    {
        return _samples;
    }

    //! Retrieve size of unit
    size_t unit(void) const
    {
        return _unit;
    }

    //! Retrieve total count of filling
    size_t total(void) const
    {
        return _total;
    }

    //! Reset the buffer
    void reset(void)
    {
        _total = 0;
        _buffer.clear();
        _buffer.reserve(_samples * _unit);
    }

    //! Fill the buffer
    void fill(const void *data)
    {
        if (_samples > 0) {
            if ((_buffer.size() / _unit) >= _samples) {
                std::uniform_int_distribution<size_t> dt(0, _total);
                size_t i = dt(_mt);

                if (i < _samples) {
                    memcpy(const_cast<char *>(_buffer.data()) + (i * _unit),
                           data, _unit);
                }
            } else {
                _buffer.append(reinterpret_cast<const char *>(data), _unit);
            }
        }
        ++_total;
    }

    //! Split the sampling result into an array
    std::vector<const void *> split(void) const
    {
        size_t count = _buffer.size() / _unit;
        std::vector<const void *> pool;
        pool.reserve(count);

        for (size_t i = 0; i < count; ++i) {
            pool.push_back(_buffer.data() + (i * _unit));
        }
        return pool;
    }

private:
    //! Disable them
    BufferSampler(void) = delete;

    //! Members
    size_t _samples;
    size_t _unit;
    size_t _total;
    std::mt19937 _mt;
    std::string _buffer;
};

} // namespace mercury

#endif // __MERCURY_UTILITY_BUFFER_SAMPLER_H__
