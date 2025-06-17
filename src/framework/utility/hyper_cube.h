/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     hyper_cube.h
 *   \author   Hechong.xyf
 *   \date     Mar 2018
 *   \version  1.0.0
 *   \brief    Interface of Mercury Utility Hyper Cube
 */

#ifndef __MERCURY_UTILITY_HYPER_CUBE_H__
#define __MERCURY_UTILITY_HYPER_CUBE_H__

#include "cube.h"
#include <string>
#include <unordered_map>

namespace mercury {

/*! Hyper Cube
 */
class HyperCube
{
public:
    //! Constructor
    HyperCube(void) : _map() {}

    //! Constructor
    HyperCube(const HyperCube &rhs) : _map(rhs._map) {}

    //! Constructor
    HyperCube(HyperCube &&rhs) : _map()
    {
        _map.swap(rhs._map);
    }

    //! Destructor
    ~HyperCube(void) {}

    //! Assignment
    HyperCube &operator=(const HyperCube &rhs)
    {
        _map = rhs._map;
        return *this;
    }

    //! Assignment
    HyperCube &operator=(HyperCube &&rhs)
    {
        _map = std::move(rhs._map);
        return *this;
    }

    //! Test if the element is exist
    bool has(const std::string &key) const
    {
        return (_map.find(key) != _map.end());
    }

    //! Emplace a key-value pair into map
    bool emplace(const std::string &key, const Cube &val)
    {
        return _map.emplace(key, val).second;
    }

    //! Emplace a key-value pair into map
    bool emplace(const std::string &key, Cube &&val)
    {
        return _map.emplace(key, std::forward<Cube>(val)).second;
    }

    //! Emplace a key-value pair into map
    bool emplace(std::string &&key, const Cube &val)
    {
        return _map.emplace(std::forward<std::string>(key), val).second;
    }

    //! Emplace a key-value pair into map
    bool emplace(std::string &&key, Cube &&val)
    {
        return _map
            .emplace(std::forward<std::string>(key), std::forward<Cube>(val))
            .second;
    }

    //! Emplace a key-value pair into map
    template <typename T>
    bool emplace(const std::string &key, const T &val)
    {
        return _map.emplace(key, Cube(val)).second;
    }

    //! Emplace a key-value pair into map
    template <typename T>
    bool emplace(const std::string &key, T &&val)
    {
        return _map.emplace(key, Cube(std::forward<T>(val))).second;
    }

    //! Emplace a key-value pair into map
    template <typename T>
    bool emplace(std::string &&key, const T &val)
    {
        return _map.emplace(std::forward<std::string>(key), Cube(val)).second;
    }

    //! Emplace a key-value pair into map
    template <typename T>
    bool emplace(std::string &&key, T &&val)
    {
        return _map
            .emplace(std::forward<std::string>(key), Cube(std::forward<T>(val)))
            .second;
    }

    //! Clear the map
    void clear(void)
    {
        _map.clear();
    }

    //! Swap the map
    void swap(HyperCube &rhs)
    {
        _map.swap(rhs._map);
    }

    //! Erase the pair via a key
    bool erase(const std::string &key)
    {
        auto iter = _map.find(key);
        if (iter != _map.end()) {
            _map.erase(iter);
            return true;
        }
        return false;
    }

    //! Retrieve the value via a key
    bool get(const std::string &key, Cube *out) const
    {
        auto iter = _map.find(key);
        if (iter != _map.end()) {
            *out = iter->second;
            return true;
        }
        return false;
    }

    //! Retrieve the value via a key
    Cube *get(const std::string &key)
    {
        auto iter = _map.find(key);
        if (iter != _map.end()) {
            return &iter->second;
        }
        return nullptr;
    }

    //! Retrieve the value via a key
    const Cube *get(const std::string &key) const
    {
        auto iter = _map.find(key);
        if (iter != _map.end()) {
            return &iter->second;
        }
        return nullptr;
    }

    //! Retrieve the value via a key
    template <typename T>
    bool get(const std::string &key, T *out) const
    {
        auto iter = _map.find(key);
        if (iter != _map.end()) {
            if (iter->second.compatible<T>()) {
                *out = iter->second.unsafe_cast<T>();
                return true;
            }
        }
        return false;
    }

    //! Retrieve the value via a key
    template <typename T>
    T &get(const std::string &key, T &def)
    {
        auto iter = _map.find(key);
        if (iter != _map.end()) {
            if (iter->second.compatible<T>()) {
                return iter->second.unsafe_cast<T>();
            }
        }
        return def;
    }

    //! Retrieve the value via a key
    template <typename T>
    const T &get(const std::string &key, const T &def) const
    {
        auto iter = _map.find(key);
        if (iter != _map.end()) {
            if (iter->second.compatible<T>()) {
                return iter->second.unsafe_cast<T>();
            }
        }
        return def;
    }

private:
    std::unordered_map<std::string, Cube> _map;
};

} // namespace mercury

#endif // __MERCURY_UTILITY_HYPER_CUBE_H__
