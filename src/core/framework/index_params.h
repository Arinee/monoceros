/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     index_params.cc
 *   \author   qiuming
 *   \date     Jan 2019
 *   \version  1.0.0
 *   \brief    Index Parameters
 */

#ifndef __MERCURY_INDEX_PARAMS_H__
#define __MERCURY_INDEX_PARAMS_H__

#include <memory>
#include "utility/hyper_cube.h"
#include "src/core/common/common.h"

MERCURY_NAMESPACE_BEGIN(core);

//! Trying compatible with T
#define _TRYING_COMPATIBLE(cube, T, out)                                       \
    if (cube->compatible<T>()) return (*out = cube->unsafe_cast<T>(), true)

//! Trying compatible with T
#define _TRYING_COMPATIBLE_FORCE(cube, T, out)                                 \
    if (cube->compatible<T>())                                                 \
    return (*out = static_cast<std::remove_pointer<decltype(out)>::type>(      \
                cube->unsafe_cast<T>()),                                       \
            true)

//! Trying compatible with T (Boolean)
#define _TRYING_COMPATIBLE_BOOL(cube, T, out)                                  \
    if (cube->compatible<T>()) return (*out = !!cube->unsafe_cast<T>(), true)

//! Trying compatible with T (String)
#define _TRYING_COMPATIBLE_STRING(cube, T, out)                                \
    if (cube->compatible<T>())                                                 \
    return (out->assign(std::to_string(cube->unsafe_cast<T>())), true)

/*! Index Params
 */
class IndexParams
{
public:
    typedef std::shared_ptr<IndexParams> Pointer;
    //! Constructor
    IndexParams(void) : _hyper_cube() {}

    //! Constructor
    IndexParams(const IndexParams &rhs) : _hyper_cube(rhs._hyper_cube) {}

    //! Constructor
    IndexParams(IndexParams &&rhs) : _hyper_cube()
    {
        _hyper_cube.swap(rhs._hyper_cube);
    }

    //! Destructor
    ~IndexParams(void) {}

    //! Assignment
    IndexParams &operator=(const IndexParams &rhs)
    {
        _hyper_cube = rhs._hyper_cube;
        return *this;
    }

    //! Assignment
    IndexParams &operator=(IndexParams &&rhs)
    {
        _hyper_cube.swap(rhs._hyper_cube);
        return *this;
    }

    //! Test if the element is exist
    bool has(const std::string &key) const
    {
        return _hyper_cube.has(key);
    }

    //! Clear the map
    void clear(void)
    {
        _hyper_cube.clear();
    }

    //! Erase the pair via a key
    bool erase(const std::string &key)
    {
        return _hyper_cube.erase(key);
    }

    //! Set the value of key in boolean
    bool set(const std::string &key, bool val)
    {
        return _hyper_cube.emplace<bool>(key, val);
    }

    //! Set the value of key in boolean
    bool set(std::string &&key, bool val)
    {
        return _hyper_cube.emplace<bool>(std::forward<std::string>(key), val);
    }

    //! Set the value of key in int8
    bool set(const std::string &key, int8_t val)
    {
        return _hyper_cube.emplace<int8_t>(key, val);
    }

    //! Set the value of key in int8
    bool set(std::string &&key, int8_t val)
    {
        return _hyper_cube.emplace<int8_t>(std::forward<std::string>(key), val);
    }

    //! Set the value of key in int16
    bool set(const std::string &key, int16_t val)
    {
        return _hyper_cube.emplace<int16_t>(key, val);
    }

    //! Set the value of key in int16
    bool set(std::string &&key, int16_t val)
    {
        return _hyper_cube.emplace<int16_t>(std::forward<std::string>(key),
                                            val);
    }

    //! Set the value of key in int32
    bool set(const std::string &key, int32_t val)
    {
        return _hyper_cube.emplace<int32_t>(key, val);
    }

    //! Set the value of key in int32
    bool set(std::string &&key, int32_t val)
    {
        return _hyper_cube.emplace<int32_t>(std::forward<std::string>(key),
                                            val);
    }

    //! Set the value of key in int64
    bool set(const std::string &key, int64_t val)
    {
        return _hyper_cube.emplace<int64_t>(key, val);
    }

    //! Set the value of key in int64
    bool set(std::string &&key, int64_t val)
    {
        return _hyper_cube.emplace<int64_t>(std::forward<std::string>(key),
                                            val);
    }

    //! Set the value of key in uint8
    bool set(const std::string &key, uint8_t val)
    {
        return _hyper_cube.emplace<uint8_t>(key, val);
    }

    //! Set the value of key in uint8
    bool set(std::string &&key, uint8_t val)
    {
        return _hyper_cube.emplace<uint8_t>(std::forward<std::string>(key),
                                            val);
    }

    //! Set the value of key in uint16
    bool set(const std::string &key, uint16_t val)
    {
        return _hyper_cube.emplace<uint16_t>(key, val);
    }

    //! Set the value of key in uint16
    bool set(std::string &&key, uint16_t val)
    {
        return _hyper_cube.emplace<uint16_t>(std::forward<std::string>(key),
                                             val);
    }

    //! Set the value of key in uint32
    bool set(const std::string &key, uint32_t val)
    {
        return _hyper_cube.emplace<uint32_t>(key, val);
    }

    //! Set the value of key in uint32
    bool set(std::string &&key, uint32_t val)
    {
        return _hyper_cube.emplace<uint32_t>(std::forward<std::string>(key),
                                             val);
    }

    //! Set the value of key in uint64
    bool set(const std::string &key, uint64_t val)
    {
        return _hyper_cube.emplace<uint64_t>(key, val);
    }

    //! Set the value of key in uint64
    bool set(std::string &&key, uint64_t val)
    {
        return _hyper_cube.emplace<uint64_t>(std::forward<std::string>(key),
                                             val);
    }

    //! Set the value of key in float
    bool set(const std::string &key, float val)
    {
        return _hyper_cube.emplace<float>(key, val);
    }

    //! Set the value of key in float
    bool set(std::string &&key, float val)
    {
        return _hyper_cube.emplace<float>(std::forward<std::string>(key), val);
    }

    //! Set the value of key in double
    bool set(const std::string &key, double val)
    {
        return _hyper_cube.emplace<double>(key, val);
    }

    //! Set the value of key in double
    bool set(std::string &&key, double val)
    {
        return _hyper_cube.emplace<double>(std::forward<std::string>(key), val);
    }

    //! Set the value of key in string
    bool set(const std::string &key, const std::string &val)
    {
        return _hyper_cube.emplace<std::string>(key, val);
    }

    //! Set the value of key in string
    bool set(const std::string &key, std::string &&val)
    {
        return _hyper_cube.emplace<std::string>(key,
                                                std::forward<std::string>(val));
    }

    //! Set the value of key in string
    bool set(std::string &&key, const std::string &val)
    {
        return _hyper_cube.emplace<std::string>(std::forward<std::string>(key),
                                                val);
    }

    //! Set the value of key in string
    bool set(std::string &&key, std::string &&val)
    {
        return _hyper_cube.emplace<std::string>(std::forward<std::string>(key),
                                                std::forward<std::string>(val));
    }

    //! Set the value of key in string
    bool set(const std::string &key, const char *val)
    {
        return _hyper_cube.emplace<std::string>(key, std::string(val));
    }

    //! Set the value of key in string
    bool set(std::string &&key, const char *val)
    {
        return _hyper_cube.emplace<std::string>(std::forward<std::string>(key),
                                                std::string(val));
    }

    //! Set the value of key in T
    template <typename T>
    bool set(const std::string &key, const T &val)
    {
        return _hyper_cube.emplace<T>(key, val);
    }

    //! Set the value of key in T
    template <typename T>
    bool set(const std::string &key, T &&val)
    {
        return _hyper_cube.emplace<T>(key, std::forward<T>(val));
    }

    //! Set the value of key in T
    template <typename T>
    bool set(std::string &&key, const T &val)
    {
        return _hyper_cube.emplace<T>(std::forward<std::string>(key), val);
    }

    //! Set the value of key in T
    template <typename T>
    bool set(std::string &&key, T &&val)
    {
        return _hyper_cube.emplace<T>(std::forward<std::string>(key),
                                      std::forward<T>(val));
    }

    //! Retrieve the value in boolean
    bool get(const std::string &key, bool *out) const
    {
        const Cube *cube = _hyper_cube.get(key);
        if (cube) {
            _TRYING_COMPATIBLE(cube, bool, out);
            _TRYING_COMPATIBLE_BOOL(cube, int8_t, out);
            _TRYING_COMPATIBLE_BOOL(cube, uint8_t, out);
            _TRYING_COMPATIBLE_BOOL(cube, int16_t, out);
            _TRYING_COMPATIBLE_BOOL(cube, uint16_t, out);
            _TRYING_COMPATIBLE_BOOL(cube, int32_t, out);
            _TRYING_COMPATIBLE_BOOL(cube, uint32_t, out);
            _TRYING_COMPATIBLE_BOOL(cube, int64_t, out);
            _TRYING_COMPATIBLE_BOOL(cube, uint64_t, out);
            _TRYING_COMPATIBLE_BOOL(cube, float, out);
            _TRYING_COMPATIBLE_BOOL(cube, double, out);
        }
        return false;
    }

    //! Retrieve the value in int8
    bool get(const std::string &key, int8_t *out) const
    {
        const Cube *cube = _hyper_cube.get(key);
        if (cube) {
            _TRYING_COMPATIBLE(cube, int8_t, out);
            _TRYING_COMPATIBLE(cube, uint8_t, out);
            _TRYING_COMPATIBLE(cube, bool, out);
            _TRYING_COMPATIBLE_FORCE(cube, uint16_t, out);
            _TRYING_COMPATIBLE_FORCE(cube, int16_t, out);
            _TRYING_COMPATIBLE_FORCE(cube, uint32_t, out);
            _TRYING_COMPATIBLE_FORCE(cube, int32_t, out);
            _TRYING_COMPATIBLE_FORCE(cube, uint64_t, out);
            _TRYING_COMPATIBLE_FORCE(cube, int64_t, out);
            _TRYING_COMPATIBLE_FORCE(cube, float, out);
            _TRYING_COMPATIBLE_FORCE(cube, double, out);
        }
        return false;
    }

    //! Retrieve the value in int16
    bool get(const std::string &key, int16_t *out) const
    {
        const Cube *cube = _hyper_cube.get(key);
        if (cube) {
            _TRYING_COMPATIBLE(cube, int16_t, out);
            _TRYING_COMPATIBLE(cube, uint16_t, out);
            _TRYING_COMPATIBLE(cube, int8_t, out);
            _TRYING_COMPATIBLE(cube, uint8_t, out);
            _TRYING_COMPATIBLE(cube, bool, out);
            _TRYING_COMPATIBLE_FORCE(cube, uint32_t, out);
            _TRYING_COMPATIBLE_FORCE(cube, int32_t, out);
            _TRYING_COMPATIBLE_FORCE(cube, uint64_t, out);
            _TRYING_COMPATIBLE_FORCE(cube, int64_t, out);
            _TRYING_COMPATIBLE_FORCE(cube, float, out);
            _TRYING_COMPATIBLE_FORCE(cube, double, out);
        }
        return false;
    }

    //! Retrieve the value in int32
    bool get(const std::string &key, int32_t *out) const
    {
        const Cube *cube = _hyper_cube.get(key);
        if (cube) {
            _TRYING_COMPATIBLE(cube, int32_t, out);
            _TRYING_COMPATIBLE(cube, uint32_t, out);
            _TRYING_COMPATIBLE(cube, int16_t, out);
            _TRYING_COMPATIBLE(cube, uint16_t, out);
            _TRYING_COMPATIBLE(cube, int8_t, out);
            _TRYING_COMPATIBLE(cube, uint8_t, out);
            _TRYING_COMPATIBLE(cube, bool, out);
            _TRYING_COMPATIBLE_FORCE(cube, uint64_t, out);
            _TRYING_COMPATIBLE_FORCE(cube, int64_t, out);
            _TRYING_COMPATIBLE_FORCE(cube, float, out);
            _TRYING_COMPATIBLE_FORCE(cube, double, out);
        }
        return false;
    }

    //! Retrieve the value in int64
    bool get(const std::string &key, int64_t *out) const
    {
        const Cube *cube = _hyper_cube.get(key);
        if (cube) {
            _TRYING_COMPATIBLE(cube, int64_t, out);
            _TRYING_COMPATIBLE(cube, uint64_t, out);
            _TRYING_COMPATIBLE(cube, int32_t, out);
            _TRYING_COMPATIBLE(cube, uint32_t, out);
            _TRYING_COMPATIBLE(cube, int16_t, out);
            _TRYING_COMPATIBLE(cube, uint16_t, out);
            _TRYING_COMPATIBLE(cube, int8_t, out);
            _TRYING_COMPATIBLE(cube, uint8_t, out);
            _TRYING_COMPATIBLE(cube, bool, out);
            _TRYING_COMPATIBLE_FORCE(cube, float, out);
            _TRYING_COMPATIBLE_FORCE(cube, double, out);
        }
        return false;
    }

    //! Retrieve the value in uint8
    bool get(const std::string &key, uint8_t *out) const
    {
        const Cube *cube = _hyper_cube.get(key);
        if (cube) {
            _TRYING_COMPATIBLE(cube, uint8_t, out);
            _TRYING_COMPATIBLE(cube, int8_t, out);
            _TRYING_COMPATIBLE(cube, bool, out);
            _TRYING_COMPATIBLE_FORCE(cube, uint16_t, out);
            _TRYING_COMPATIBLE_FORCE(cube, int16_t, out);
            _TRYING_COMPATIBLE_FORCE(cube, uint32_t, out);
            _TRYING_COMPATIBLE_FORCE(cube, int32_t, out);
            _TRYING_COMPATIBLE_FORCE(cube, uint64_t, out);
            _TRYING_COMPATIBLE_FORCE(cube, int64_t, out);
            _TRYING_COMPATIBLE_FORCE(cube, float, out);
            _TRYING_COMPATIBLE_FORCE(cube, double, out);
        }
        return false;
    }

    //! Retrieve the value in uint16
    bool get(const std::string &key, uint16_t *out) const
    {
        const Cube *cube = _hyper_cube.get(key);
        if (cube) {
            _TRYING_COMPATIBLE(cube, uint16_t, out);
            _TRYING_COMPATIBLE(cube, int16_t, out);
            _TRYING_COMPATIBLE(cube, uint8_t, out);
            _TRYING_COMPATIBLE(cube, int8_t, out);
            _TRYING_COMPATIBLE(cube, bool, out);
            _TRYING_COMPATIBLE_FORCE(cube, uint32_t, out);
            _TRYING_COMPATIBLE_FORCE(cube, int32_t, out);
            _TRYING_COMPATIBLE_FORCE(cube, uint64_t, out);
            _TRYING_COMPATIBLE_FORCE(cube, int64_t, out);
            _TRYING_COMPATIBLE_FORCE(cube, float, out);
            _TRYING_COMPATIBLE_FORCE(cube, double, out);
        }
        return false;
    }

    //! Retrieve the value in uint32
    bool get(const std::string &key, uint32_t *out) const
    {
        const Cube *cube = _hyper_cube.get(key);
        if (cube) {
            _TRYING_COMPATIBLE(cube, uint32_t, out);
            _TRYING_COMPATIBLE(cube, int32_t, out);
            _TRYING_COMPATIBLE(cube, uint16_t, out);
            _TRYING_COMPATIBLE(cube, int16_t, out);
            _TRYING_COMPATIBLE(cube, uint8_t, out);
            _TRYING_COMPATIBLE(cube, int8_t, out);
            _TRYING_COMPATIBLE(cube, bool, out);
            _TRYING_COMPATIBLE_FORCE(cube, uint64_t, out);
            _TRYING_COMPATIBLE_FORCE(cube, int64_t, out);
            _TRYING_COMPATIBLE_FORCE(cube, float, out);
            _TRYING_COMPATIBLE_FORCE(cube, double, out);
        }
        return false;
    }

    //! Retrieve the value in uint64
    bool get(const std::string &key, uint64_t *out) const
    {
        const Cube *cube = _hyper_cube.get(key);
        if (cube) {
            _TRYING_COMPATIBLE(cube, uint64_t, out);
            _TRYING_COMPATIBLE(cube, int64_t, out);
            _TRYING_COMPATIBLE(cube, uint32_t, out);
            _TRYING_COMPATIBLE(cube, int32_t, out);
            _TRYING_COMPATIBLE(cube, uint16_t, out);
            _TRYING_COMPATIBLE(cube, int16_t, out);
            _TRYING_COMPATIBLE(cube, uint8_t, out);
            _TRYING_COMPATIBLE(cube, int8_t, out);
            _TRYING_COMPATIBLE(cube, bool, out);
            _TRYING_COMPATIBLE_FORCE(cube, float, out);
            _TRYING_COMPATIBLE_FORCE(cube, double, out);
        }
        return false;
    }

    //! Retrieve the value in float
    bool get(const std::string &key, float *out) const
    {
        const Cube *cube = _hyper_cube.get(key);
        if (cube) {
            _TRYING_COMPATIBLE(cube, float, out);
            _TRYING_COMPATIBLE(cube, uint64_t, out);
            _TRYING_COMPATIBLE(cube, int64_t, out);
            _TRYING_COMPATIBLE(cube, uint32_t, out);
            _TRYING_COMPATIBLE(cube, int32_t, out);
            _TRYING_COMPATIBLE(cube, uint16_t, out);
            _TRYING_COMPATIBLE(cube, int16_t, out);
            _TRYING_COMPATIBLE(cube, uint8_t, out);
            _TRYING_COMPATIBLE(cube, int8_t, out);
            _TRYING_COMPATIBLE(cube, bool, out);
            _TRYING_COMPATIBLE_FORCE(cube, double, out);
        }
        return false;
    }

    //! Retrieve the value in double
    bool get(const std::string &key, double *out) const
    {
        const Cube *cube = _hyper_cube.get(key);
        if (cube) {
            _TRYING_COMPATIBLE(cube, double, out);
            _TRYING_COMPATIBLE(cube, float, out);
            _TRYING_COMPATIBLE(cube, uint64_t, out);
            _TRYING_COMPATIBLE(cube, int64_t, out);
            _TRYING_COMPATIBLE(cube, uint32_t, out);
            _TRYING_COMPATIBLE(cube, int32_t, out);
            _TRYING_COMPATIBLE(cube, uint16_t, out);
            _TRYING_COMPATIBLE(cube, int16_t, out);
            _TRYING_COMPATIBLE(cube, uint8_t, out);
            _TRYING_COMPATIBLE(cube, int8_t, out);
            _TRYING_COMPATIBLE(cube, bool, out);
        }
        return false;
    }

    //! Retrieve the value in string
    bool get(const std::string &key, std::string *out) const
    {
        const Cube *cube = _hyper_cube.get(key);
        if (cube) {
            _TRYING_COMPATIBLE(cube, std::string, out);
            _TRYING_COMPATIBLE_STRING(cube, uint64_t, out);
            _TRYING_COMPATIBLE_STRING(cube, int64_t, out);
            _TRYING_COMPATIBLE_STRING(cube, uint32_t, out);
            _TRYING_COMPATIBLE_STRING(cube, int32_t, out);
            _TRYING_COMPATIBLE_STRING(cube, uint16_t, out);
            _TRYING_COMPATIBLE_STRING(cube, int16_t, out);
            _TRYING_COMPATIBLE_STRING(cube, uint8_t, out);
            _TRYING_COMPATIBLE_STRING(cube, int8_t, out);
            _TRYING_COMPATIBLE_STRING(cube, bool, out);
            _TRYING_COMPATIBLE_STRING(cube, float, out);
            _TRYING_COMPATIBLE_STRING(cube, double, out);
        }
        return false;
    }

    //! Retrieve the value in T
    template <typename T>
    bool get(const std::string &key, T *out) const
    {
        const Cube *cube = _hyper_cube.get(key);
        if (cube) {
            _TRYING_COMPATIBLE(cube, T, out);
        }
        return false;
    }

    //! Retrieve the value in boolean
    bool getBool(const std::string &key) const
    {
        bool result = false;
        this->get(key, &result);
        return result;
    }

    //! Retrieve the value in int8
    int8_t getInt8(const std::string &key) const
    {
        int8_t result = 0;
        this->get(key, &result);
        return result;
    }

    //! Retrieve the value in int16
    int16_t getInt16(const std::string &key) const
    {
        int16_t result = 0;
        this->get(key, &result);
        return result;
    }

    //! Retrieve the value in int32
    int32_t getInt32(const std::string &key) const
    {
        int32_t result = 0;
        this->get(key, &result);
        return result;
    }

    //! Retrieve the value in int64
    int64_t getInt64(const std::string &key) const
    {
        int64_t result = 0;
        this->get(key, &result);
        return result;
    }

    //! Retrieve the value in uint8
    uint8_t getUint8(const std::string &key) const
    {
        uint8_t result = 0;
        this->get(key, &result);
        return result;
    }

    //! Retrieve the value in uint16
    uint16_t getUint16(const std::string &key) const
    {
        uint16_t result = 0;
        this->get(key, &result);
        return result;
    }

    //! Retrieve the value in uint32
    uint32_t getUint32(const std::string &key) const
    {
        uint32_t result = 0;
        this->get(key, &result);
        return result;
    }

    //! Retrieve the value in uint64
    uint64_t getUint64(const std::string &key) const
    {
        uint64_t result = 0;
        this->get(key, &result);
        return result;
    }

    //! Retrieve the value in float
    float getFloat(const std::string &key) const
    {
        float result = 0.0f;
        this->get(key, &result);
        return result;
    }

    //! Retrieve the value in double
    double getDouble(const std::string &key) const
    {
        double result = 0.0f;
        this->get(key, &result);
        return result;
    }

    //! Retrieve the value in string
    std::string getString(const std::string &key) const
    {
        std::string result;
        this->get(key, &result);
        return result;
    }

    //! Retrieve the value in T
    template <typename T>
    T *getObject(const std::string &key)
    {
        Cube *cube = _hyper_cube.get(key);
        if (cube) {
            if (cube->compatible<T>()) {
                return &(cube->unsafe_cast<T>());
            }
        }
        return nullptr;
    }

    //! Retrieve the value in T
    template <typename T>
    const T *getObject(const std::string &key) const
    {
        const Cube *cube = _hyper_cube.get(key);
        if (cube) {
            if (cube->compatible<T>()) {
                return &(cube->unsafe_cast<T>());
            }
        }
        return nullptr;
    }

    //! Update parameters from OS environment
    void updateFromEnvironment(void);

private:
    HyperCube _hyper_cube;
};

#undef _TRYING_COMPATIBLE
#undef _TRYING_COMPATIBLE_FORCE
#undef _TRYING_COMPATIBLE_BOOL
#undef _TRYING_COMPATIBLE_STRING

//! Empty parameters
extern const IndexParams EmptyParams;

MERCURY_NAMESPACE_END(core);

#endif // __MERCURY_INDEX_PARAMS_H__

