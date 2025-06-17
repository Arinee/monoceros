/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     mod_json_plus.h
 *   \author   Rainvan (Yunfeng.Xiao)
 *   \date     Dev 2012
 *   \version  1.0.0
 *   \brief    Interface of JSON Parser/Generator (C++)
 */

#ifndef __MERCURY_UTILITY_JSON_MOD_JSON_PLUS_H__
#define __MERCURY_UTILITY_JSON_MOD_JSON_PLUS_H__

#include "mod_json.h"
#include <cfloat>
#include <cstring>
#include <stdexcept>
#include <string>

namespace mercury {

/*! JSON String
 */
class JsonString
{
public:
    typedef mod_json_size_t size_type;
    typedef mod_json_ssize_t ssize_type;
    typedef mod_json_float_t float_type;
    typedef mod_json_integer_t integer_type;

    //! Constructor
    JsonString(void) : _str(0) {}

    //! Constructor
    JsonString(const JsonString &rhs) : _str(0)
    {
        if (rhs._str) {
            _str = mod_json_string_grab(rhs._str);
        }
    }

    //! Constructor
    JsonString(const char *cstr)
    {
        _str = mod_json_string_set(cstr, cstr ? std::strlen(cstr) : 0);
    }

    //! Constructor
    JsonString(const char *cstr, size_type len)
    {
        _str = mod_json_string_set(cstr, len);
    }

    //! Constructor
    JsonString(const std::string &str)
    {
        _str = mod_json_string_set(str.c_str(), str.size());
    }

    //! Destructor
    ~JsonString(void)
    {
        mod_json_string_unset(_str);
    }

    //! Assign new contents to the string, replacing its current content
    JsonString &operator=(const JsonString &rhs)
    {
        this->assign(rhs);
        return *this;
    }

    //! Assign new contents to the string, replacing its current content
    JsonString &operator=(const char *cstr)
    {
        this->assign(cstr);
        return *this;
    }

    //! Assign new contents to the string, replacing its current content
    JsonString &operator=(const std::string &rhs)
    {
        this->assign(rhs);
        return *this;
    }

    //! Append a JSON string
    JsonString &operator+=(const JsonString &str)
    {
        this->append(str);
        return *this;
    }

    //! Append a c-style string
    JsonString &operator+=(const char *cstr)
    {
        this->append(cstr);
        return *this;
    }

    //! Append a character to string
    JsonString &operator+=(char c)
    {
        this->append(c);
        return *this;
    }

    //! Equality
    bool operator==(const JsonString &rhs) const
    {
        return (mod_json_string_compare(_str, rhs._str) == 0);
    }

    //! No equality
    bool operator!=(const JsonString &rhs) const
    {
        return !(*this == rhs);
    }

    //! Retrieve the character at index n
    char &operator[](size_type n)
    {
        if (!copyAndLeak()) {
            throw std::runtime_error("JsonString::operator[]");
        }
        return *(_str->first + n);
    }

    //! Retrieve the character at index n
    const char &operator[](size_type n) const
    {
        return *(_str->first + n);
    }

    //! Retrieve non-zero if the string is valid
    bool isValid(void) const
    {
        return (_str != (mod_json_string_t *)0);
    }

    //! Retrieve non-zero if the string is empty
    bool empty(void) const
    {
        return mod_json_string_empty(_str);
    }

    //! Assign a JSON string
    void assign(const JsonString &rhs)
    {
        mod_json_string_unset(_str);
        _str = rhs._str ? mod_json_string_grab(rhs._str) : 0;
    }

    //! Assign a c-style string
    void assign(const char *cstr)
    {
        if (cstr) {
            if (!copyOnWrite() ||
                mod_json_string_assign(_str, cstr, std::strlen(cstr)) != 0) {
                throw std::runtime_error("JsonString::assign");
            }
        }
    }

    //! Assign a c-style string
    void assign(const char *cstr, size_type len)
    {
        if (!copyOnWrite() || mod_json_string_assign(_str, cstr, len) != 0) {
            throw std::runtime_error("JsonString::assign");
        }
    }

    //! Assign a STL-style string
    void assign(const std::string &str)
    {
        if (!copyOnWrite() ||
            mod_json_string_assign(_str, str.c_str(), str.size()) != 0) {
            throw std::runtime_error("JsonString::assign");
        }
    }

    //! Append a JSON string
    void append(const JsonString &str)
    {
        if (str._str) {
            if (!copyOnWrite() || mod_json_string_add(_str, str._str) != 0) {
                throw std::runtime_error("JsonString::append");
            }
        }
    }

    //! Append a c-style string
    void append(const char *cstr)
    {
        if (cstr) {
            if (!copyOnWrite() ||
                mod_json_string_append(_str, cstr, std::strlen(cstr)) != 0) {
                throw std::runtime_error("JsonString::append");
            }
        }
    }

    //! Append a c-style string
    void append(const char *cstr, size_type len)
    {
        if (!copyOnWrite() || mod_json_string_append(_str, cstr, len) != 0) {
            throw std::runtime_error("JsonString::append");
        }
    }

    //! Append a STL-style string
    void append(const std::string &str)
    {
        if (!copyOnWrite() ||
            mod_json_string_append(_str, str.c_str(), str.size()) != 0) {
            throw std::runtime_error("JsonString::append");
        }
    }

    //! Append a character to string
    void append(char c)
    {
        if (!copyOnWrite() || mod_json_string_append(_str, &c, 1) != 0) {
            throw std::runtime_error("JsonString::append");
        }
    }

    //! Retrieve the character at index n
    char &at(size_type n)
    {
        if (this->size() <= n) {
            throw std::out_of_range("JsonString::at");
        }
        if (!copyAndLeak()) {
            throw std::runtime_error("JsonString::at");
        }
        return *(_str->first + n);
    }

    //! Retrieve the character at index n
    const char &at(size_type n) const
    {
        if (this->size() <= n) {
            throw std::out_of_range("JsonString::at");
        }
        return *(_str->first + n);
    }

    //! Request a change in capacity
    void reserve(size_type n)
    {
        if (!copyOnWrite() || mod_json_string_reserve(_str, n) != 0) {
            throw std::runtime_error("JsonString::reserve");
        }
    }

    //! Clear the JSON string
    void clear(void)
    {
        mod_json_string_unset(_str);
        _str = 0;
    }

    //! Exchange the content with another JSON string
    void swap(JsonString &rhs)
    {
        mod_json_string_t *str = _str;
        _str = rhs._str;
        rhs._str = str;
    }

    //! Retrieve the data pointer
    char *data(void)
    {
        return mod_json_string_data(_str);
    }

    //! Retrieve the data pointer
    const char *data(void) const
    {
        return mod_json_string_data(_str);
    }

    //! Retrieve HASH of a JSON string
    size_type hash(void) const
    {
        return mod_json_string_hash(_str);
    }

    //! Compare two JSON strings (case sensitive)
    int compare(const JsonString &rhs) const
    {
        return mod_json_string_compare(_str, rhs._str);
    }

    //! Compare two strings (case sensitive)
    int compare(const char *cstr) const
    {
        const char *self = this->c_str();
        if (self && cstr) {
            return std::strcmp(self, cstr);
        }

        // particular case
        if (!self && cstr) {
            return -1;
        } else if (self && !cstr) {
            return 1;
        }
        return 0;
    }

    // Encode a JSON string
    JsonString encode(void) const
    {
        JsonString ret;
        ret._str = mod_json_string_encode(_str);
        return ret;
    }

    // Decode a JSON string
    JsonString decode(void) const
    {
        JsonString ret;
        ret._str = mod_json_string_decode(_str);
        return ret;
    }

    //! Retrieve the capacity of string
    size_type capacity(void) const
    {
        return mod_json_string_capacity(_str);
    }

    //! Retrieve the length of string
    size_type size(void) const
    {
        return mod_json_string_length(_str);
    }

    //! Retrieve the length of string
    size_type length(void) const
    {
        return mod_json_string_length(_str);
    }

    //! Retrieve refer-counter of string
    ssize_type refer(void) const
    {
        return mod_json_string_refer(_str);
    }

    //! Retrieve the c-style string
    const char *c_str(void) const
    {
        return mod_json_string_cstr(_str);
    }

    //! Convert string to float
    float_type asFloat(void) const
    {
        return mod_json_string_float(_str);
    }

    //! Convert string to integer
    integer_type asInteger(void) const
    {
        return mod_json_string_integer(_str);
    }

    //! Retrieve string as a STL string
    std::string asStlString(void) const
    {
        if (!this->empty()) {
            return std::string(this->data(), this->size());
        }
        return std::string();
    }

protected:
    //! Clone the string for writing
    bool copyOnWrite(void)
    {
        if (_str) {
            if (mod_json_string_is_shared(_str)) {
                (void)mod_json_string_put(_str);
                _str = mod_json_string_clone(_str);
            }
        } else {
            _str = mod_json_string_set("", 0);
        }
        return (_str != 0);
    }

    //! Clone the value and leak it
    bool copyAndLeak(void)
    {
        if (copyOnWrite()) {
            mod_json_string_set_leaked(_str);
            return true;
        }
        return false;
    }

private:
    mod_json_string_t *_str;
};

class JsonArray;
class JsonObject;

/*! JSON Value
 */
class JsonValue
{
public:
    typedef mod_json_size_t size_type;
    typedef mod_json_ssize_t ssize_type;
    typedef mod_json_float_t float_type;
    typedef mod_json_integer_t integer_type;

    //! Constructor
    JsonValue(void) : _val(0) {}

    //! Constructor
    explicit JsonValue(const bool &val)
    {
        _val = mod_json_value_set_boolean((mod_json_boolean_t)val);
    }

    //! Constructor
    explicit JsonValue(const char &val)
    {
        _val = mod_json_value_set_integer((mod_json_integer_t)val);
    }

    //! Constructor
    explicit JsonValue(const short &val)
    {
        _val = mod_json_value_set_integer((mod_json_integer_t)val);
    }

    //! Constructor
    explicit JsonValue(const int &val)
    {
        _val = mod_json_value_set_integer((mod_json_integer_t)val);
    }

    //! Constructor
    explicit JsonValue(const long &val)
    {
        _val = mod_json_value_set_integer((mod_json_integer_t)val);
    }

    //! Constructor
    explicit JsonValue(const long long &val)
    {
        _val = mod_json_value_set_integer((mod_json_integer_t)val);
    }

    //! Constructor
    explicit JsonValue(const float &val)
    {
        _val = mod_json_value_set_float((mod_json_float_t)val);
    }

    //! Constructor
    explicit JsonValue(const double &val)
    {
        _val = mod_json_value_set_float((mod_json_float_t)val);
    }

    //! Constructor
    explicit JsonValue(const unsigned char &val)
    {
        _val = mod_json_value_set_integer((mod_json_integer_t)val);
    }

    //! Constructor
    explicit JsonValue(const unsigned short &val)
    {
        _val = mod_json_value_set_integer((mod_json_integer_t)val);
    }

    //! Constructor
    explicit JsonValue(const unsigned int &val)
    {
        _val = mod_json_value_set_integer((mod_json_integer_t)val);
    }

    //! Constructor
    explicit JsonValue(const unsigned long &val)
    {
        _val = mod_json_value_set_integer((mod_json_integer_t)val);
    }

    //! Constructor
    explicit JsonValue(const unsigned long long &val)
    {
        _val = mod_json_value_set_integer((mod_json_integer_t)val);
    }

    //! Constructor
    JsonValue(const JsonString &val)
    {
        _val = mod_json_value_set_string(*(mod_json_string_t **)&val);
    }

    //! Constructor
    JsonValue(const char *val)
    {
        _val = mod_json_value_set_buffer(val, val ? std::strlen(val) : 0);
    }

    //! Constructor
    JsonValue(const char *val, size_type len)
    {
        _val = mod_json_value_set_buffer(val, len);
    }

    //! Constructor
    JsonValue(const std::string &val)
    {
        _val = mod_json_value_set_buffer(val.data(), val.size());
    }

    //! Constructor
    JsonValue(const JsonArray &val)
    {
        _val = mod_json_value_set_array(*(mod_json_array_t **)&val);
    }

    //! Constructor
    JsonValue(const JsonObject &val)
    {
        _val = mod_json_value_set_object(*(mod_json_object_t **)&val);
    }

    //! Constructor
    JsonValue(const JsonValue &rhs) : _val(0)
    {
        if (rhs._val) {
            _val = mod_json_value_grab(rhs._val);
        }
    }

    //! Destructor
    ~JsonValue(void)
    {
        mod_json_value_unset(_val);
    }

    //! Assign new contents to the value, replacing its current content
    JsonValue &operator=(const JsonValue &rhs)
    {
        this->assign(rhs);
        return *this;
    }

    //! Assign new contents to the value, replacing its current content
    JsonValue &operator=(const bool &val)
    {
        this->assign(val);
        return *this;
    }

    //! Assign new contents to the value, replacing its current content
    JsonValue &operator=(const char &val)
    {
        this->assign(val);
        return *this;
    }

    //! Assign new contents to the value, replacing its current content
    JsonValue &operator=(const short &val)
    {
        this->assign(val);
        return *this;
    }

    //! Assign new contents to the value, replacing its current content
    JsonValue &operator=(const int &val)
    {
        this->assign(val);
        return *this;
    }

    //! Assign new contents to the value, replacing its current content
    JsonValue &operator=(const long &val)
    {
        this->assign(val);
        return *this;
    }

    //! Assign new contents to the value, replacing its current content
    JsonValue &operator=(const long long &val)
    {
        this->assign(val);
        return *this;
    }

    //! Assign new contents to the value, replacing its current content
    JsonValue &operator=(const float &val)
    {
        this->assign(val);
        return *this;
    }

    //! Assign new contents to the value, replacing its current content
    JsonValue &operator=(const double &val)
    {
        this->assign(val);
        return *this;
    }

    //! Assign new contents to the value, replacing its current content
    JsonValue &operator=(const unsigned char &val)
    {
        this->assign(val);
        return *this;
    }

    //! Assign new contents to the value, replacing its current content
    JsonValue &operator=(const unsigned short &val)
    {
        this->assign(val);
        return *this;
    }

    //! Assign new contents to the value, replacing its current content
    JsonValue &operator=(const unsigned int &val)
    {
        this->assign(val);
        return *this;
    }

    //! Assign new contents to the value, replacing its current content
    JsonValue &operator=(const unsigned long &val)
    {
        this->assign(val);
        return *this;
    }

    //! Assign new contents to the value, replacing its current content
    JsonValue &operator=(const unsigned long long &val)
    {
        this->assign(val);
        return *this;
    }

    //! Assign new contents to the value, replacing its current content
    JsonValue &operator=(const JsonString &val)
    {
        this->assign(val);
        return *this;
    }

    //! Assign new contents to the value, replacing its current content
    JsonValue &operator=(const char *val)
    {
        this->assign(val);
        return *this;
    }

    //! Assign new contents to the value, replacing its current content
    JsonValue &operator=(const std::string &val)
    {
        this->assign(val);
        return *this;
    }

    //! Assign new contents to the value, replacing its current content
    JsonValue &operator=(const JsonArray &arr)
    {
        this->assign(arr);
        return *this;
    }

    //! Assign new contents to the value, replacing its current content
    JsonValue &operator=(const JsonObject &obj)
    {
        this->assign(obj);
        return *this;
    }

    //! Equality
    bool operator==(const JsonValue &rhs) const
    {
        return mod_json_value_is_equal(_val, rhs._val);
    }

    //! No equality
    bool operator!=(const JsonValue &rhs) const
    {
        return !(*this == rhs);
    }

    //! Treat self value as object by force, retrieving value of a key
    JsonValue &operator[](const char *key)
    {
        return this->getValue(key);
    }

    //! Retrieve a reference of value by a key
    const JsonValue &operator[](const char *key) const
    {
        return this->getValue(key);
    }

    //! Treat self value as object by force, retrieving value of a key
    JsonValue &operator[](const JsonString &key)
    {
        return this->getValue(key.c_str());
    }

    //! Retrieve a reference of value by a key
    const JsonValue &operator[](const JsonString &key) const
    {
        return this->getValue(key.c_str());
    }

    //! Treat self value as array by force, retrieving value at index n
    JsonValue &operator[](size_type n)
    {
        return this->getValue(n);
    }

    //! Retrieve a reference of value at index n
    const JsonValue &operator[](size_type n) const
    {
        return this->getValue(n);
    }

    //! Retrieve non-zero if the value is valid
    bool isValid(void) const
    {
        return (_val != (mod_json_value_t *)0);
    }

    //! Retrieve non-zero if the value is a object
    bool isObject(void) const
    {
        return mod_json_value_is_object(_val);
    }

    //! Retrieve non-zero if the value is an array
    bool isArray(void) const
    {
        return mod_json_value_is_array(_val);
    }

    //! Retrieve non-zero if the value is a string
    bool isString(void) const
    {
        return mod_json_value_is_string(_val);
    }

    //! Retrieve non-zero if the value is null
    bool isNull(void) const
    {
        return mod_json_value_is_null(_val);
    }

    //! Retrieve non-zero if the value is a float
    bool isFloat(void) const
    {
        return mod_json_value_is_float(_val);
    }

    //! Retrieve non-zero if the value is an integer
    bool isInteger(void) const
    {
        return mod_json_value_is_integer(_val);
    }

    //! Retrieve non-zero if the value is a boolean
    bool isBoolean(void) const
    {
        return mod_json_value_is_boolean(_val);
    }

    //! Assign new contents to the value, replacing its current content
    void assign(const JsonValue &rhs)
    {
        mod_json_value_unset(_val);
        _val = rhs._val ? mod_json_value_grab(rhs._val) : 0;
    }

    //! Assign new contents to the value, replacing its current content
    void assign(const bool &val)
    {
        if (!copyOnWrite()) {
            throw std::runtime_error("JsonValue::assign");
        }
        mod_json_value_assign_boolean(_val, (mod_json_boolean_t)val);
    }

    //! Assign new contents to the value, replacing its current content
    void assign(const char &val)
    {
        if (!copyOnWrite()) {
            throw std::runtime_error("JsonValue::assign");
        }
        mod_json_value_assign_integer(_val, (mod_json_integer_t)val);
    }

    //! Assign new contents to the value, replacing its current content
    void assign(const short &val)
    {
        if (!copyOnWrite()) {
            throw std::runtime_error("JsonValue::assign");
        }
        mod_json_value_assign_integer(_val, (mod_json_integer_t)val);
    }

    //! Assign new contents to the value, replacing its current content
    void assign(const int &val)
    {
        if (!copyOnWrite()) {
            throw std::runtime_error("JsonValue::assign");
        }
        mod_json_value_assign_integer(_val, (mod_json_integer_t)val);
    }

    //! Assign new contents to the value, replacing its current content
    void assign(const long &val)
    {
        if (!copyOnWrite()) {
            throw std::runtime_error("JsonValue::assign");
        }
        mod_json_value_assign_integer(_val, (mod_json_integer_t)val);
    }

    //! Assign new contents to the value, replacing its current content
    void assign(const long long &val)
    {
        if (!copyOnWrite()) {
            throw std::runtime_error("JsonValue::assign");
        }
        mod_json_value_assign_integer(_val, (mod_json_integer_t)val);
    }

    //! Assign new contents to the value, replacing its current content
    void assign(const float &val)
    {
        if (!copyOnWrite()) {
            throw std::runtime_error("JsonValue::assign");
        }
        mod_json_value_assign_float(_val, (mod_json_float_t)val);
    }

    //! Assign new contents to the value, replacing its current content
    void assign(const double &val)
    {
        if (!copyOnWrite()) {
            throw std::runtime_error("JsonValue::assign");
        }
        mod_json_value_assign_float(_val, (mod_json_float_t)val);
    }

    //! Assign new contents to the value, replacing its current content
    void assign(const unsigned char &val)
    {
        if (!copyOnWrite()) {
            throw std::runtime_error("JsonValue::assign");
        }
        mod_json_value_assign_integer(_val, (mod_json_integer_t)val);
    }

    //! Assign new contents to the value, replacing its current content
    void assign(const unsigned short &val)
    {
        if (!copyOnWrite()) {
            throw std::runtime_error("JsonValue::assign");
        }
        mod_json_value_assign_integer(_val, (mod_json_integer_t)val);
    }

    //! Assign new contents to the value, replacing its current content
    void assign(const unsigned int &val)
    {
        if (!copyOnWrite()) {
            throw std::runtime_error("JsonValue::assign");
        }
        mod_json_value_assign_integer(_val, (mod_json_integer_t)val);
    }

    //! Assign new contents to the value, replacing its current content
    void assign(const unsigned long &val)
    {
        if (!copyOnWrite()) {
            throw std::runtime_error("JsonValue::assign");
        }
        mod_json_value_assign_integer(_val, (mod_json_integer_t)val);
    }

    //! Assign new contents to the value, replacing its current content
    void assign(const unsigned long long &val)
    {
        if (!copyOnWrite()) {
            throw std::runtime_error("JsonValue::assign");
        }
        mod_json_value_assign_integer(_val, (mod_json_integer_t)val);
    }

    //! Assign new contents to the value, replacing its current content
    void assign(const JsonString &val)
    {
        if (!copyOnWrite()) {
            throw std::runtime_error("JsonValue::assign");
        }
        mod_json_value_assign_string(_val, *(mod_json_string_t **)&val);
    }

    //! Assign new contents to the value, replacing its current content
    void assign(const char *val)
    {
        JsonString str(val);
        if (!str.isValid() || !copyOnWrite()) {
            throw std::runtime_error("JsonValue::assign");
        }
        mod_json_value_assign_string(_val, *(mod_json_string_t **)&str);
    }

    //! Assign new contents to the value, replacing its current content
    void assign(const char *val, size_type len)
    {
        JsonString str(val, len);
        if (!str.isValid() || !copyOnWrite()) {
            throw std::runtime_error("JsonValue::assign");
        }
        mod_json_value_assign_string(_val, *(mod_json_string_t **)&str);
    }

    //! Assign new contents to the value, replacing its current content
    void assign(const std::string &val)
    {
        JsonString str(val);
        if (!str.isValid() || !copyOnWrite()) {
            throw std::runtime_error("JsonValue::assign");
        }
        mod_json_value_assign_string(_val, *(mod_json_string_t **)&str);
    }

    //! Assign new contents to the value, replacing its current content
    void assign(const JsonArray &arr);

    //! Assign new contents to the value, replacing its current content
    void assign(const JsonObject &obj);

    //! Retrieve refer-counter of JSON value
    ssize_type refer(void) const
    {
        return mod_json_value_refer(_val);
    }

    //! Retrieve value as JSON format string
    JsonString asJsonString(void) const
    {
        mod_json_string_t *tmp = mod_json_dump(_val);
        JsonString ret = *reinterpret_cast<JsonString *>(&tmp);
        if (tmp) {
            mod_json_string_unset(tmp);
        }
        return ret;
    }

    //! Retrieve value as a STL string
    std::string asStlString(void) const
    {
        if (isString()) {
            return toString().asStlString();
        }
        return std::string();
    }

    //! Retrieve value as JSON string
    const JsonString &asString(void) const
    {
        if (!isString()) {
            throw std::logic_error("JsonValue::asString");
        }
        return toString();
    }

    //! Retrieve value as c-style string
    const char *asCString(void) const
    {
        return mod_json_value_cstring(_val);
    }

    //! Retrieve value as JSON string
    JsonString &asString(void)
    {
        if (!isString()) {
            throw std::logic_error("JsonValue::asString");
        }
        if (!copyAndLeak()) {
            throw std::runtime_error("JsonValue::asString");
        }
        return toString();
    }

    //! Retrieve value as JSON array
    const JsonArray &asArray(void) const
    {
        if (!isArray()) {
            throw std::logic_error("JsonValue::asArray");
        }
        return toArray();
    }

    //! Retrieve value as JSON array
    JsonArray &asArray(void)
    {
        if (!isArray()) {
            throw std::logic_error("JsonValue::asArray");
        }
        if (!copyAndLeak()) {
            throw std::runtime_error("JsonValue::asArray");
        }
        return toArray();
    }

    //! Retrieve value as JSON object
    const JsonObject &asObject(void) const
    {
        if (!isObject()) {
            throw std::logic_error("JsonValue::asObject");
        }
        return toObject();
    }

    //! Retrieve value as JSON object
    JsonObject &asObject(void)
    {
        if (!isObject()) {
            throw std::logic_error("JsonValue::asObject");
        }
        if (!copyAndLeak()) {
            throw std::runtime_error("JsonValue::asObject");
        }
        return toObject();
    }

    //! Retrieve value as float
    float_type asFloat(void) const
    {
        return mod_json_value_float(_val);
    }

    //! Retrieve value as integer
    integer_type asInteger(void) const
    {
        return mod_json_value_integer(_val);
    }

    //! Retrieve value as boolean
    bool asBool(void) const
    {
        return mod_json_value_boolean(_val);
    }

    //! Exchange the content with another JSON value
    void swap(JsonValue &rhs)
    {
        mod_json_value_t *val = _val;
        _val = rhs._val;
        rhs._val = val;
    }

    //! Merge another JSON value
    void merge(const JsonValue &rhs)
    {
        if (!copyOnWrite()) {
            throw std::runtime_error("JsonValue::merge");
        }
        mod_json_value_merge(_val, rhs._val);
    }

    //! Parse a sting as a JSON value
    bool parse(const char *str)
    {
        mod_json_token_t *tok = mod_json_token_create(NULL);

        if (tok) {
            mod_json_value_t *jval = mod_json_parse(tok, str);

            mod_json_token_destroy(tok);
            if (jval) {
                *this = *reinterpret_cast<JsonValue *>(&jval);
                mod_json_value_unset(jval);
                return isValid();
            }
        }
        return false;
    }

    //! Parse a sting as a JSON value
    bool parse(const JsonString &str)
    {
        return this->parse(str.c_str());
    }

    //! Parse a sting as a JSON value
    bool parse(const std::string &str)
    {
        return this->parse(str.c_str());
    }

    //! Retrieve reference of a invalid JSON value
    static const JsonValue &Invalid(void)
    {
        return _invalid;
    }

protected:
    //! Clone the value for writing
    bool copyOnWrite(void)
    {
        if (_val) {
            if (mod_json_value_is_shared(_val)) {
                (void)mod_json_value_put(_val);
                _val = mod_json_value_clone(_val);
            }
        } else {
            _val = mod_json_value_set_null();
        }
        return (_val != 0);
    }

    //! Clone the value and leak it
    bool copyAndLeak(void)
    {
        if (copyOnWrite()) {
            mod_json_value_set_leaked(_val);
            return true;
        }
        return false;
    }

    //! Convert value to JSON object
    JsonObject &toObject(void);

    //! Convert value to JSON object
    const JsonObject &toObject(void) const;

    //! Convert value to JSON array
    JsonArray &toArray(void);

    //! Convert value to JSON array
    const JsonArray &toArray(void) const;

    //! Convert value to JSON string
    JsonString &toString(void);

    //! Convert value to JSON string
    const JsonString &toString(void) const;

    //! Treat self value as object by force, retrieving value of a key
    JsonValue &getValue(const char *key);

    //! Retrieve a reference of value by a key
    const JsonValue &getValue(const char *key) const;

    //! Treat self value as array by force, retrieving value at index n
    JsonValue &getValue(size_type n);

    //! Retrieve a reference of value at index n
    const JsonValue &getValue(size_type n) const;

    //! Set the new array to the value, replacing its current content
    void setValue(const JsonArray &val);

    //! Set the new object to the value, replacing its current content
    void setValue(const JsonObject &val);

protected:
    static const JsonValue _invalid;

private:
    mod_json_value_t *_val;
};

/*! JSON Array
 */
class JsonArray
{
public:
    typedef mod_json_size_t size_type;
    typedef mod_json_ssize_t ssize_type;

    class iterator;
    class const_iterator;
    class reverse_iterator;
    class const_reverse_iterator;

    /*! Const iterator of JSON Array
     */
    class const_iterator
    {
        friend class JsonArray;
        friend class JsonArray::iterator;
        friend class JsonArray::reverse_iterator;
        friend class JsonArray::const_reverse_iterator;

    public:
        //! Constructor
        const_iterator(void) : _iter(0) {}

        //! Equality
        bool operator==(const const_iterator &rhs) const
        {
            return (_iter == rhs._iter);
        }

        //! No equality
        bool operator!=(const const_iterator &rhs) const
        {
            return (_iter != rhs._iter);
        }

        //! Increment (Prefix)
        const_iterator &operator++()
        {
            ++_iter;
            return *this;
        }

        //! Increment (Suffix)
        const_iterator operator++(int)
        {
            const_iterator tmp = *this;
            ++_iter;
            return tmp;
        }

        //! Decrement (Prefix)
        const_iterator &operator--()
        {
            --_iter;
            return *this;
        }

        //! Decrement (Suffix)
        const_iterator operator--(int)
        {
            const_iterator tmp = *this;
            --_iter;
            return tmp;
        }

        //! Indirection (eg. *iter)
        const JsonValue &operator*() const
        {
            return *reinterpret_cast<const JsonValue *>(_iter);
        }

        //! Structure dereference (eg. iter->)
        const JsonValue *operator->() const
        {
            return reinterpret_cast<const JsonValue *>(_iter);
        }

        //! Retrieve as const reverse iterator
        operator const_reverse_iterator() const
        {
            return const_reverse_iterator(_iter);
        }

    protected:
        //! Constructor for friends
        const_iterator(mod_json_value_t *const *iter) : _iter(iter) {}

    private:
        mod_json_value_t *const *_iter;
    };

    /*! iterator of JSON Array
     */
    class iterator
    {
        friend class JsonArray;
        friend class JsonArray::reverse_iterator;

    public:
        //! Constructor
        iterator(void) : _iter(0) {}

        //! Equality
        bool operator==(const iterator &rhs) const
        {
            return (_iter == rhs._iter);
        }

        //! No equality
        bool operator!=(const iterator &rhs) const
        {
            return (_iter != rhs._iter);
        }

        //! Increment (Prefix)
        iterator &operator++()
        {
            ++_iter;
            return *this;
        }

        //! Increment (Suffix)
        iterator operator++(int)
        {
            iterator tmp = *this;
            ++_iter;
            return tmp;
        }

        //! Decrement (Prefix)
        iterator &operator--()
        {
            --_iter;
            return *this;
        }

        //! Decrement (Suffix)
        iterator operator--(int)
        {
            iterator tmp = *this;
            --_iter;
            return tmp;
        }

        //! Indirection (eg. *iter)
        JsonValue &operator*() const
        {
            return *reinterpret_cast<JsonValue *>(_iter);
        }

        //! Structure dereference (eg. iter->)
        JsonValue *operator->() const
        {
            return reinterpret_cast<JsonValue *>(_iter);
        }

        //! Retrieve as const iterator
        operator const_iterator() const
        {
            return const_iterator(_iter);
        }

        //! Retrieve as reverse iterator
        operator reverse_iterator() const
        {
            return reverse_iterator(_iter);
        }

        //! Retrieve as const reverse iterator
        operator const_reverse_iterator() const
        {
            return const_reverse_iterator(_iter);
        }

    protected:
        //! Constructor for friends
        iterator(mod_json_value_t **iter) : _iter(iter) {}

    private:
        mod_json_value_t **_iter;
    };

    /*! Const Reverse iterator of JSON Array
     */
    class const_reverse_iterator
    {
        friend class JsonArray;
        friend class JsonArray::iterator;
        friend class JsonArray::const_iterator;
        friend class JsonArray::reverse_iterator;

    public:
        //! Constructor
        const_reverse_iterator(void) : _iter(0) {}

        //! Equality
        bool operator==(const const_reverse_iterator &rhs) const
        {
            return (_iter == rhs._iter);
        }

        //! No equality
        bool operator!=(const const_reverse_iterator &rhs) const
        {
            return (_iter != rhs._iter);
        }

        //! Increment (Prefix)
        const_reverse_iterator &operator++()
        {
            --_iter;
            return *this;
        }

        //! Increment (Suffix)
        const_reverse_iterator operator++(int)
        {
            const_reverse_iterator tmp = *this;
            --_iter;
            return tmp;
        }

        //! Decrement (Prefix)
        const_reverse_iterator &operator--()
        {
            ++_iter;
            return *this;
        }

        //! Decrement (Suffix)
        const_reverse_iterator operator--(int)
        {
            const_reverse_iterator tmp = *this;
            ++_iter;
            return tmp;
        }

        //! Indirection (eg. *iter)
        const JsonValue &operator*() const
        {
            return *reinterpret_cast<const JsonValue *>(_iter);
        }

        //! Structure dereference (eg. iter->)
        const JsonValue *operator->() const
        {
            return reinterpret_cast<const JsonValue *>(_iter);
        }

        //! Retrieve as const iterator
        operator const_iterator() const
        {
            return const_iterator(_iter);
        }

    protected:
        //! Constructor for friends
        const_reverse_iterator(mod_json_value_t *const *iter) : _iter(iter) {}

    private:
        mod_json_value_t *const *_iter;
    };

    /*! Reverse iterator of JSON Array
     */
    class reverse_iterator
    {
        friend class JsonArray;
        friend class JsonArray::iterator;

    public:
        //! Constructor
        reverse_iterator(void) : _iter(0) {}

        //! Equality
        bool operator==(const reverse_iterator &rhs) const
        {
            return (_iter == rhs._iter);
        }

        //! No equality
        bool operator!=(const reverse_iterator &rhs) const
        {
            return (_iter != rhs._iter);
        }

        //! Increment (Prefix)
        reverse_iterator &operator++()
        {
            --_iter;
            return *this;
        }

        //! Increment (Suffix)
        reverse_iterator operator++(int)
        {
            reverse_iterator tmp = *this;
            --_iter;
            return tmp;
        }

        //! Decrement (Prefix)
        reverse_iterator &operator--()
        {
            ++_iter;
            return *this;
        }

        //! Decrement (Suffix)
        reverse_iterator operator--(int)
        {
            reverse_iterator tmp = *this;
            ++_iter;
            return tmp;
        }

        //! Indirection (eg. *iter)
        JsonValue &operator*() const
        {
            return *reinterpret_cast<JsonValue *>(_iter);
        }

        //! Structure dereference (eg. iter->)
        JsonValue *operator->() const
        {
            return reinterpret_cast<JsonValue *>(_iter);
        }

        //! Retrieve as iterator
        operator iterator() const
        {
            return iterator(_iter);
        }

        //! Retrieve as const iterator
        operator const_iterator() const
        {
            return const_iterator(_iter);
        }

        //! Retrieve as const reverse iterator
        operator const_reverse_iterator() const
        {
            return const_reverse_iterator(_iter);
        }

    protected:
        //! Constructor for friends
        reverse_iterator(mod_json_value_t **iter) : _iter(iter) {}

    private:
        mod_json_value_t **_iter;
    };

    //! Constructor
    JsonArray(void) : _arr(0) {}

    //! Constructor
    JsonArray(const JsonArray &rhs) : _arr(0)
    {
        if (rhs._arr) {
            _arr = mod_json_array_grab(rhs._arr);
        }
    }

    //! Destructor
    ~JsonArray(void)
    {
        mod_json_array_unset(_arr);
    }

    //! Assign new contents to the array, replacing its current content
    JsonArray &operator=(const JsonArray &rhs)
    {
        this->assign(rhs);
        return *this;
    }

    //! Equality
    bool operator==(const JsonArray &rhs) const
    {
        return mod_json_array_is_equal(_arr, rhs._arr);
    }

    //! No equality
    bool operator!=(const JsonArray &rhs) const
    {
        return !(*this == rhs);
    }

    //! Retrieve the value at index n, if no one exists, throw an exception.
    JsonValue &operator[](size_type n)
    {
        return this->at(n);
    }

    //! Retrieve the value at index n, if no one exists, return a null value.
    const JsonValue &operator[](size_type n) const
    {
        return ((n < this->size()) ? this->getValue(n) : JsonValue::Invalid());
    }

    //! Retrieve non-zero if the array is valid
    bool isValid(void) const
    {
        return (_arr != (mod_json_array_t *)0);
    }

    //! Retrieve non-zero if the array is empty
    bool empty(void) const
    {
        return mod_json_array_empty(_arr);
    }

    //! Retrieve the size of JSON array
    size_type size(void) const
    {
        return mod_json_array_count(_arr);
    }

    //! Retrieve the capacity of JSON array
    size_type capacity(void) const
    {
        return mod_json_array_capacity(_arr);
    }

    //! Retrieve refer-counter of JSON array
    ssize_type refer(void) const
    {
        return mod_json_array_refer(_arr);
    }

    //! Assign new contents to the array, replacing its current content
    void assign(const JsonArray &rhs)
    {
        mod_json_array_unset(_arr);
        _arr = rhs._arr ? mod_json_array_grab(rhs._arr) : 0;
    }

    //! Request a change in capacity
    void reserve(size_type n)
    {
        if (!copyOnWrite() || mod_json_array_reserve(_arr, n) != 0) {
            throw std::runtime_error("JsonArray::reserve");
        }
    }

    //! Reverse the order of the elements
    void reverse(void)
    {
        if (_arr && copyOnWrite()) {
            mod_json_array_reverse(_arr);
        }
    }

    //! Push a value to array
    void push(const JsonValue &val)
    {
        JsonValue tmp(val);

        if (!copyOnWrite() ||
            mod_json_array_push(_arr, *((mod_json_value_t **)&tmp)) != 0) {
            throw std::runtime_error("JsonArray::push");
        }
    }

    //! Pop the last element from array
    void pop(void)
    {
        if (_arr) {
            if (!copyOnWrite()) {
                throw std::runtime_error("JsonArray::pop");
            }
            mod_json_array_pop(_arr);
        }
    }

    //! Remove the first element of array
    void shift(void)
    {
        if (_arr) {
            if (!copyOnWrite()) {
                throw std::runtime_error("JsonArray::shift");
            }
            mod_json_array_shift(_arr);
        }
    }

    //! Retrieve the value at index n
    JsonValue &at(size_type n)
    {
        if (this->size() <= n) {
            throw std::out_of_range("JsonArray::at");
        }
        if (!copyAndLeak()) {
            throw std::runtime_error("JsonArray::at");
        }
        return this->getValue(n);
    }

    //! Retrieve the value at index n
    const JsonValue &at(size_type n) const
    {
        if (this->size() <= n) {
            throw std::out_of_range("JsonArray::at");
        }
        return this->getValue(n);
    }

    //! Retrieve a reference to the first element
    JsonValue &front(void)
    {
        if (this->size() <= 0) {
            throw std::out_of_range("JsonArray::front");
        }
        if (!copyAndLeak()) {
            throw std::runtime_error("JsonArray::front");
        }
        return this->getValue(0);
    }

    //! Retrieve a reference to the first element
    const JsonValue &front(void) const
    {
        if (this->size() <= 0) {
            throw std::out_of_range("JsonArray::front");
        }
        return this->getValue(0);
    }

    //! Retrieve a reference to the last element
    JsonValue &back(void)
    {
        if (this->size() <= 0) {
            throw std::out_of_range("JsonArray::back");
        }
        if (!copyAndLeak()) {
            throw std::runtime_error("JsonArray::back");
        }
        return this->getValue(this->size() - 1);
    }

    //! Retrieve a reference to the last element
    const JsonValue &back(void) const
    {
        if (this->size() <= 0) {
            throw std::out_of_range("JsonArray::back");
        }
        return this->getValue(this->size() - 1);
    }

    //! Clear the JSON array
    void clear(void)
    {
        mod_json_array_unset(_arr);
        _arr = 0;
    }

    //! Exchange the content with another JSON array
    void swap(JsonArray &rhs)
    {
        mod_json_array_t *arr = _arr;
        _arr = rhs._arr;
        rhs._arr = arr;
    }

    //! Merge another JSON array
    void merge(const JsonArray &rhs)
    {
        if (!copyOnWrite()) {
            throw std::runtime_error("JsonArray::merge");
        }
        mod_json_array_merge(_arr, rhs._arr);
    }

    //! Resize a JSON array so that it contains n elements
    void resize(size_type n, const JsonValue &val = JsonValue())
    {
        if (!copyOnWrite() ||
            mod_json_array_resize(_arr, n, *((mod_json_value_t **)&val)) != 0) {
            throw std::runtime_error("JsonArray::resize");
        }
    }

    //! Retrieve an iterator pointing to the first element
    iterator begin(void)
    {
        if (copyAndLeak()) {
            return iterator(mod_json_array_begin(_arr));
        }
        return iterator();
    }

    //! Retrieve a const iterator pointing to the first element
    const_iterator begin(void) const
    {
        if (_arr) {
            return const_iterator(mod_json_array_begin(_arr));
        }
        return const_iterator();
    }

    //! Retrieve a const iterator pointing to the first element
    const_iterator cbegin(void) const
    {
        if (_arr) {
            return const_iterator(mod_json_array_begin(_arr));
        }
        return const_iterator();
    }

    //! Retrieve a reverse iterator pointing to the last element
    reverse_iterator rbegin(void)
    {
        if (copyAndLeak()) {
            return reverse_iterator(mod_json_array_rbegin(_arr));
        }
        return reverse_iterator();
    }

    //! Retrieve a const reverse iterator pointing to the last element
    const_reverse_iterator rbegin(void) const
    {
        if (_arr) {
            return const_reverse_iterator(mod_json_array_rbegin(_arr));
        }
        return const_reverse_iterator();
    }

    //! Retrieve a const reverse iterator pointing to the last element
    const_reverse_iterator crbegin(void) const
    {
        if (_arr) {
            return const_reverse_iterator(mod_json_array_rbegin(_arr));
        }
        return const_reverse_iterator();
    }

    //! Retrieve an iterator pointing to the past-the-end element
    iterator end(void)
    {
        if (copyAndLeak()) {
            return iterator(mod_json_array_end(_arr));
        }
        return iterator();
    }

    //! Retrieve a const iterator pointing to the past-the-end element
    const_iterator end(void) const
    {
        if (_arr) {
            return const_iterator(mod_json_array_end(_arr));
        }
        return const_iterator();
    }

    //! Retrieve a const iterator pointing to the past-the-end element
    const_iterator cend(void) const
    {
        if (_arr) {
            return const_iterator(mod_json_array_end(_arr));
        }
        return const_iterator();
    }

    //! Retrieve a reverse pointing to the past-the-end element
    reverse_iterator rend(void)
    {
        if (copyAndLeak()) {
            return reverse_iterator(mod_json_array_rend(_arr));
        }
        return reverse_iterator();
    }

    //! Retrieve a const reverse pointing to the past-the-end element
    const_reverse_iterator rend(void) const
    {
        if (_arr) {
            return const_reverse_iterator(mod_json_array_rend(_arr));
        }
        return const_reverse_iterator();
    }

    //! Retrieve a const reverse pointing to the past-the-end element
    const_reverse_iterator crend(void) const
    {
        if (_arr) {
            return const_reverse_iterator(mod_json_array_rend(_arr));
        }
        return const_reverse_iterator();
    }

protected:
    //! Clone the array for writing
    bool copyOnWrite(void)
    {
        if (_arr) {
            if (mod_json_array_is_shared(_arr)) {
                (void)mod_json_array_put(_arr);
                _arr = mod_json_array_clone(_arr);
            }
        } else {
            _arr = mod_json_array_set_default();
        }
        return (_arr != 0);
    }

    //! Clone the array and leak it
    bool copyAndLeak(void)
    {
        if (copyOnWrite()) {
            mod_json_array_set_leaked(_arr);
            return true;
        }
        return false;
    }

    //! Retrieve the value at index n
    JsonValue &getValue(size_type n)
    {
        return *reinterpret_cast<JsonValue *>(_arr->first + n);
    }

    //! Retrieve the value at index n
    const JsonValue &getValue(size_type n) const
    {
        return *reinterpret_cast<JsonValue *>(_arr->first + n);
    }

private:
    mod_json_array_t *_arr;
};

/*! JSON Pair
 */
class JsonPair
{
    friend class JsonObject;

public:
    //! Constructor
    JsonPair(void) : _pair(0) {}

    //! Retrieve non-zero if the pair is valid
    bool isValid(void) const
    {
        return (_pair != (mod_json_pair_t *)0);
    }

    //! Retrieve the key of pair
    const JsonString &key(void) const
    {
        return *reinterpret_cast<JsonString *>(&_pair->key);
    }

    //! Retrieve the value of pair
    JsonValue &value(void)
    {
        return *reinterpret_cast<JsonValue *>(&_pair->val);
    }

    //! Retrieve the value of pair
    const JsonValue &value(void) const
    {
        return *reinterpret_cast<JsonValue *>(&_pair->val);
    }

protected:
    //! Constructor for friends
    JsonPair(mod_json_pair_t *pair) : _pair(pair) {}

private:
    mod_json_pair_t *_pair;
};

/*! JSON Object
 */
class JsonObject
{
public:
    typedef mod_json_size_t size_type;
    typedef mod_json_ssize_t ssize_type;

    class iterator;
    class const_iterator;
    class reverse_iterator;
    class const_reverse_iterator;

    /*! Const iterator of JSON Object
     */
    class const_iterator
    {
        friend class JsonObject;
        friend class JsonObject::iterator;
        friend class JsonObject::reverse_iterator;
        friend class JsonObject::const_reverse_iterator;

    public:
        //! Constructor
        const_iterator(void) : _iter(0) {}

        //! Equality
        bool operator==(const const_iterator &rhs) const
        {
            return (_iter == rhs._iter);
        }

        //! No equality
        bool operator!=(const const_iterator &rhs) const
        {
            return (_iter != rhs._iter);
        }

        //! Increment (Prefix)
        const_iterator &operator++()
        {
            ++_iter;
            return *this;
        }

        //! Increment (Suffix)
        const_iterator operator++(int)
        {
            const_iterator tmp = *this;
            ++_iter;
            return tmp;
        }

        //! Decrement (Prefix)
        const_iterator &operator--()
        {
            --_iter;
            return *this;
        }

        //! Decrement (Suffix)
        const_iterator operator--(int)
        {
            const_iterator tmp = *this;
            --_iter;
            return tmp;
        }

        //! Indirection (eg. *iter)
        const JsonPair &operator*() const
        {
            return *reinterpret_cast<const JsonPair *>(&_iter);
        }

        //! Structure dereference (eg. iter->)
        const JsonPair *operator->() const
        {
            return reinterpret_cast<const JsonPair *>(&_iter);
        }

        //! Retrieve as const reverse iterator
        operator const_reverse_iterator() const
        {
            return const_reverse_iterator(_iter);
        }

    protected:
        //! Constructor for friends
        const_iterator(const mod_json_pair_t *iter) : _iter(iter) {}

    private:
        const mod_json_pair_t *_iter;
    };

    /*! iterator of JSON Object
     */
    class iterator
    {
        friend class JsonObject;
        friend class JsonObject::reverse_iterator;

    public:
        //! Constructor
        iterator(void) : _iter(0) {}

        //! Equality
        bool operator==(const iterator &rhs) const
        {
            return (_iter == rhs._iter);
        }

        //! No equality
        bool operator!=(const iterator &rhs) const
        {
            return (_iter != rhs._iter);
        }

        //! Increment (Prefix)
        iterator &operator++()
        {
            ++_iter;
            return *this;
        }

        //! Increment (Suffix)
        iterator operator++(int)
        {
            iterator tmp = *this;
            ++_iter;
            return tmp;
        }

        //! Decrement (Prefix)
        iterator &operator--()
        {
            --_iter;
            return *this;
        }

        //! Decrement (Suffix)
        iterator operator--(int)
        {
            iterator tmp = *this;
            --_iter;
            return tmp;
        }

        //! Indirection (eg. *iter)
        JsonPair &operator*() const
        {
            return *reinterpret_cast<JsonPair *>((mod_json_pair_t **)&_iter);
        }

        //! Structure dereference (eg. iter->)
        JsonPair *operator->() const
        {
            return reinterpret_cast<JsonPair *>((mod_json_pair_t **)&_iter);
        }

        //! Retrieve as const iterator
        operator const_iterator() const
        {
            return const_iterator(_iter);
        }

        //! Retrieve as reverse iterator
        operator reverse_iterator() const
        {
            return reverse_iterator(_iter);
        }

        //! Retrieve as const reverse iterator
        operator const_reverse_iterator() const
        {
            return const_reverse_iterator(_iter);
        }

    protected:
        //! Constructor for friends
        iterator(mod_json_pair_t *iter) : _iter(iter) {}

    private:
        mod_json_pair_t *_iter;
    };

    /*! Const Reverse iterator of JSON Object
     */
    class const_reverse_iterator
    {
        friend class JsonObject;
        friend class JsonObject::iterator;
        friend class JsonObject::const_iterator;
        friend class JsonObject::reverse_iterator;

    public:
        //! Constructor
        const_reverse_iterator(void) : _iter(0) {}

        //! Equality
        bool operator==(const const_reverse_iterator &rhs) const
        {
            return (_iter == rhs._iter);
        }

        //! No equality
        bool operator!=(const const_reverse_iterator &rhs) const
        {
            return (_iter != rhs._iter);
        }

        //! Increment (Prefix)
        const_reverse_iterator &operator++()
        {
            --_iter;
            return *this;
        }

        //! Increment (Suffix)
        const_reverse_iterator operator++(int)
        {
            const_reverse_iterator tmp = *this;
            --_iter;
            return tmp;
        }

        //! Decrement (Prefix)
        const_reverse_iterator &operator--()
        {
            ++_iter;
            return *this;
        }

        //! Decrement (Suffix)
        const_reverse_iterator operator--(int)
        {
            const_reverse_iterator tmp = *this;
            ++_iter;
            return tmp;
        }

        //! Indirection (eg. *iter)
        const JsonPair &operator*() const
        {
            return *reinterpret_cast<const JsonPair *>(&_iter);
        }

        //! Structure dereference (eg. iter->)
        const JsonPair *operator->() const
        {
            return reinterpret_cast<const JsonPair *>(&_iter);
        }

        //! Retrieve as const iterator
        operator const_iterator() const
        {
            return const_iterator(_iter);
        }

    protected:
        //! Constructor for friends
        const_reverse_iterator(const mod_json_pair_t *iter) : _iter(iter) {}

    private:
        const mod_json_pair_t *_iter;
    };

    /*! iterator of JSON Object
     */
    class reverse_iterator
    {
        friend class JsonObject;
        friend class JsonArray::iterator;

    public:
        //! Constructor
        reverse_iterator(void) : _iter(0) {}

        //! Equality
        bool operator==(const reverse_iterator &rhs) const
        {
            return (_iter == rhs._iter);
        }

        //! No equality
        bool operator!=(const reverse_iterator &rhs) const
        {
            return (_iter != rhs._iter);
        }

        //! Increment (Prefix)
        reverse_iterator &operator++()
        {
            --_iter;
            return *this;
        }

        //! Increment (Suffix)
        reverse_iterator operator++(int)
        {
            reverse_iterator tmp = *this;
            --_iter;
            return tmp;
        }

        //! Decrement (Prefix)
        reverse_iterator &operator--()
        {
            ++_iter;
            return *this;
        }

        //! Decrement (Suffix)
        reverse_iterator operator--(int)
        {
            reverse_iterator tmp = *this;
            ++_iter;
            return tmp;
        }

        //! Indirection (eg. *iter)
        JsonPair &operator*() const
        {
            return *reinterpret_cast<JsonPair *>((mod_json_pair_t **)&_iter);
        }

        //! Structure dereference (eg. iter->)
        JsonPair *operator->() const
        {
            return reinterpret_cast<JsonPair *>((mod_json_pair_t **)&_iter);
        }

        //! Retrieve as iterator
        operator iterator() const
        {
            return iterator(_iter);
        }

        //! Retrieve as const iterator
        operator const_iterator() const
        {
            return const_iterator(_iter);
        }

        //! Retrieve as const reverse iterator
        operator const_reverse_iterator() const
        {
            return const_reverse_iterator(_iter);
        }

    protected:
        //! Constructor for friends
        reverse_iterator(mod_json_pair_t *iter) : _iter(iter) {}

    private:
        mod_json_pair_t *_iter;
    };

    //! Constructor
    JsonObject(void) : _obj(0) {}

    //! Constructor
    JsonObject(const JsonObject &rhs) : _obj(0)
    {
        if (rhs._obj) {
            _obj = mod_json_object_grab(rhs._obj);
        }
    }

    //! Destructor
    ~JsonObject(void)
    {
        mod_json_object_unset(_obj);
    }

    //! Assign new contents to the object, replacing its current content
    JsonObject &operator=(const JsonObject &rhs)
    {
        this->assign(rhs);
        return *this;
    }

    //! Equality
    bool operator==(const JsonObject &rhs) const
    {
        return mod_json_object_is_equal(_obj, rhs._obj);
    }

    //! No equality
    bool operator!=(const JsonObject &rhs) const
    {
        return !(*this == rhs);
    }

    //! Retrieve the value of a key, if no one exists, create a new one.
    JsonValue &operator[](const char *key)
    {
        if (!key) {
            throw std::invalid_argument("JsonObject::operator[]");
        }

        if (!copyAndLeak()) {
            throw std::runtime_error("JsonObject::operator[]");
        }

        JsonPair pair(mod_json_object_touch(_obj, key));
        if (!pair.isValid()) {
            throw std::runtime_error("JsonObject::operator[]");
        }
        return pair.value();
    }

    //! Retrieve the value of a key, if no one exists, return a null value.
    const JsonValue &operator[](const char *key) const
    {
        if (!key) {
            throw std::invalid_argument("JsonObject::operator[]");
        }

        JsonPair pair(mod_json_object_find(_obj, key));
        return (pair.isValid() ? pair.value() : JsonValue::Invalid());
    }

    //! Retrieve the value of a key, if no one exists, create a new one.
    JsonValue &operator[](const JsonString &key)
    {
        return (*this)[key.c_str()];
    }

    //! Retrieve the value of a key, if no one exists, return a null value.
    const JsonValue &operator[](const JsonString &key) const
    {
        return (*this)[key.c_str()];
    }

    //! Retrieve non-zero if the object is valid
    bool isValid(void) const
    {
        return (_obj != (mod_json_object_t *)0);
    }

    //! Retrieve non-zero if the object is empty
    bool empty(void) const
    {
        return mod_json_object_empty(_obj);
    }

    //! Retrieve the size of JSON object
    size_type size(void) const
    {
        return mod_json_object_count(_obj);
    }

    //! Retrieve refer-counter of JSON object
    ssize_type refer(void) const
    {
        return mod_json_object_refer(_obj);
    }

    //! Assign new contents to the object, replacing its current content
    void assign(const JsonObject &rhs)
    {
        mod_json_object_unset(_obj);
        _obj = rhs._obj ? mod_json_object_grab(rhs._obj) : 0;
    }

    //! Clear the JSON object
    void clear(void)
    {
        mod_json_object_unset(_obj);
        _obj = 0;
    }

    //! Set the value of a key
    bool set(const char *key, const JsonValue &val)
    {
        return this->set(JsonString(key), val);
    }

    //! Set the value of a key
    bool set(const JsonString &key, const JsonValue &val)
    {
        JsonValue tmp(val);

        if (!copyOnWrite()) {
            throw std::runtime_error("JsonObject::set");
        }
        return (mod_json_object_insert(_obj, *(mod_json_string_t **)&key,
                                       *(mod_json_value_t **)&tmp) !=
                (mod_json_pair_t *)0);
    }

    //! Retrieve the value of a key
    JsonValue &get(const char *key, JsonValue &def)
    {
        if (!copyAndLeak()) {
            throw std::runtime_error("JsonObject::get");
        }

        JsonPair pair(mod_json_object_find(_obj, key));
        if (pair.isValid()) {
            return pair.value();
        }
        return def;
    }

    //! Retrieve the value of a key
    JsonValue &get(const JsonString &key, JsonValue &def)
    {
        return this->get(key.c_str(), def);
    }

    //! Retrieve the value of a key
    const JsonValue &get(const char *key, const JsonValue &def) const
    {
        const JsonPair pair(mod_json_object_find(_obj, key));
        return (pair.isValid() ? pair.value() : def);
    }

    //! Retrieve the value of a key
    const JsonValue &get(const JsonString &key, const JsonValue &def) const
    {
        return this->get(key.c_str(), def);
    }

    //! Delete a key-value pair from JSON object
    void unset(const char *key)
    {
        if (_obj && key) {
            if (!copyOnWrite()) {
                throw std::runtime_error("JsonObject::unset");
            }
            mod_json_object_erase(_obj, key);
        }
    }

    //! Retrieve non-zero if the key exists in JSON object
    bool has(const char *key) const
    {
        return (mod_json_object_find(_obj, key) != (mod_json_pair_t *)0);
    }

    //! Exchange the content with another JSON object
    void swap(JsonObject &rhs)
    {
        mod_json_object_t *obj = _obj;
        _obj = rhs._obj;
        rhs._obj = obj;
    }

    //! Merge another JSON object
    void merge(const JsonObject &rhs)
    {
        if (!copyOnWrite()) {
            throw std::runtime_error("JsonObject::merge");
        }
        mod_json_object_merge(_obj, rhs._obj);
    }

    //! Retrieve an iterator pointing to the first element
    iterator begin(void)
    {
        if (copyAndLeak()) {
            return iterator(mod_json_object_begin(_obj));
        }
        return iterator();
    }

    //! Retrieve a const iterator pointing to the first element
    const_iterator begin(void) const
    {
        if (_obj) {
            return const_iterator(mod_json_object_begin(_obj));
        }
        return const_iterator();
    }

    //! Retrieve a const iterator pointing to the first element
    const_iterator cbegin(void) const
    {
        if (_obj) {
            return const_iterator(mod_json_object_begin(_obj));
        }
        return const_iterator();
    }

    //! Retrieve a reverse iterator pointing to the last element
    reverse_iterator rbegin(void)
    {
        if (copyAndLeak()) {
            return reverse_iterator(mod_json_object_rbegin(_obj));
        }
        return reverse_iterator();
    }

    //! Retrieve a const reverse iterator pointing to the last element
    const_reverse_iterator rbegin(void) const
    {
        if (_obj) {
            return const_reverse_iterator(mod_json_object_rbegin(_obj));
        }
        return const_reverse_iterator();
    }

    //! Retrieve a const reverse iterator pointing to the last element
    const_reverse_iterator crbegin(void) const
    {
        if (_obj) {
            return const_reverse_iterator(mod_json_object_rbegin(_obj));
        }
        return const_reverse_iterator();
    }

    //! Retrieve an iterator pointing to the past-the-end element
    iterator end(void)
    {
        if (copyAndLeak()) {
            return iterator(mod_json_object_end(_obj));
        }
        return iterator();
    }

    //! Retrieve a const iterator pointing to the past-the-end element
    const_iterator end(void) const
    {
        if (_obj) {
            return const_iterator(mod_json_object_end(_obj));
        }
        return const_iterator();
    }

    //! Retrieve a const iterator pointing to the past-the-end element
    const_iterator cend(void) const
    {
        if (_obj) {
            return const_iterator(mod_json_object_end(_obj));
        }
        return const_iterator();
    }

    //! Retrieve a reverse pointing to the past-the-end element
    reverse_iterator rend(void)
    {
        if (copyAndLeak()) {
            return reverse_iterator(mod_json_object_rend(_obj));
        }
        return reverse_iterator();
    }

    //! Retrieve a const reverse pointing to the past-the-end element
    const_reverse_iterator rend(void) const
    {
        if (_obj) {
            return const_reverse_iterator(mod_json_object_rend(_obj));
        }
        return const_reverse_iterator();
    }

    //! Retrieve a const reverse pointing to the past-the-end element
    const_reverse_iterator crend(void) const
    {
        if (_obj) {
            return const_reverse_iterator(mod_json_object_rend(_obj));
        }
        return const_reverse_iterator();
    }

protected:
    //! Clone the object for writing
    bool copyOnWrite(void)
    {
        if (_obj) {
            if (mod_json_object_is_shared(_obj)) {
                (void)mod_json_object_put(_obj);
                _obj = mod_json_object_clone(_obj);
            }
        } else {
            _obj = mod_json_object_set_default();
        }
        return (_obj != 0);
    }

    //! Clone the object and leak it
    bool copyAndLeak(void)
    {
        if (copyOnWrite()) {
            mod_json_object_set_leaked(_obj);
            return true;
        }
        return false;
    }

private:
    mod_json_object_t *_obj;
};

//! Assign new contents to the value, replacing its current content
inline void JsonValue::assign(const JsonArray &arr)
{
    this->setValue(arr);
}

//! Assign new contents to the value, replacing its current content
inline void JsonValue::assign(const JsonObject &obj)
{
    this->setValue(obj);
}

//! Convert value to JSON object
inline JsonObject &JsonValue::toObject(void)
{
    return *reinterpret_cast<JsonObject *>(&_val->data.c_obj);
}

//! Convert value to JSON object
inline const JsonObject &JsonValue::toObject(void) const
{
    return *reinterpret_cast<JsonObject *>(&_val->data.c_obj);
}

//! Convert value to JSON array
inline JsonArray &JsonValue::toArray(void)
{
    return *reinterpret_cast<JsonArray *>(&_val->data.c_arr);
}

//! Convert value to JSON array
inline const JsonArray &JsonValue::toArray(void) const
{
    return *reinterpret_cast<JsonArray *>(&_val->data.c_arr);
}

//! Convert value to JSON string
inline JsonString &JsonValue::toString(void)
{
    return *reinterpret_cast<JsonString *>(&_val->data.c_str);
}

//! Convert value to JSON string
inline const JsonString &JsonValue::toString(void) const
{
    return *reinterpret_cast<JsonString *>(&_val->data.c_str);
}

//! Treat self value as object by force, retrieving value of a key
inline JsonValue &JsonValue::getValue(const char *key)
{
    if (!isObject()) {
        *this = JsonObject();
    }
    if (!copyAndLeak()) {
        throw std::runtime_error("JsonValue::getValue");
    }
    return (toObject())[key];
}

//! Retrieve a reference of value by a key
inline const JsonValue &JsonValue::getValue(const char *key) const
{
    return (isObject() ? (toObject())[key] : JsonValue::Invalid());
}

//! Treat self value as array by force, retrieving value at index n
inline JsonValue &JsonValue::getValue(size_type n)
{
    if (!isArray()) {
        throw std::logic_error("JsonValue::getValue");
    }
    if (!copyAndLeak()) {
        throw std::runtime_error("JsonValue::getValue");
    }
    return (toArray())[n];
}

//! Retrieve a reference of value at index n
inline const JsonValue &JsonValue::getValue(size_type n) const
{
    return (isArray() ? (toArray())[n] : JsonValue::Invalid());
}

//! Set the new array to the value, replacing its current content
inline void JsonValue::setValue(const JsonArray &val)
{
    if (!copyOnWrite()) {
        throw std::runtime_error("JsonValue::setValue");
    }
    mod_json_value_assign_array(_val, *(mod_json_array_t **)&val);
}

//! Set the new object to the value, replacing its current content
inline void JsonValue::setValue(const JsonObject &val)
{
    if (!copyOnWrite()) {
        throw std::runtime_error("JsonValue::setValue");
    }
    mod_json_value_assign_object(_val, *(mod_json_object_t **)&val);
}

/*! JSON Parser
 */
class JsonParser
{
public:
    typedef mod_json_size_t size_type;

    //! Constructor
    JsonParser(void)
        : _state(mod_json_state_null), _error(mod_json_error_null), _context(0)
    {
        _option.options = 0;
        _option.object_depth = 0;
        _option.array_depth = 0;
    }

    //! Destructor
    ~JsonParser(void) {}

    //! Set the max object depth
    void setObjectDepth(size_type depth)
    {
        _option.object_depth = depth;
    }

    //! Set the max array depth
    void setArrayDepth(size_type depth)
    {
        _option.array_depth = depth;
    }

    //! Enable/Disable comments
    void setComment(bool enable = true)
    {
        if (enable) {
            _option.options |= MOD_JSON_COMMENT;
        } else {
            _option.options &= ~MOD_JSON_COMMENT;
        }
    }

    //! Enable/Disable loose strings
    void setUnstrict(bool enable = true)
    {
        if (enable) {
            _option.options |= MOD_JSON_UNSTRICT;
        } else {
            _option.options &= ~MOD_JSON_UNSTRICT;
        }
    }

    //! Enable/Disable simple format
    void setSimple(bool enable = true)
    {
        if (enable) {
            _option.options |= MOD_JSON_SIMPLE;
        } else {
            _option.options &= ~MOD_JSON_SIMPLE;
        }
    }

    //! Enable/Disable single quotes support
    void setSquote(bool enable = true)
    {
        if (enable) {
            _option.options |= MOD_JSON_SQUOTE;
        } else {
            _option.options &= ~MOD_JSON_SQUOTE;
        }
    }

    //! Convert a sting to a JSON value
    bool parse(const char *str, JsonValue *out)
    {
        mod_json_token_t *tok;

        _state = mod_json_state_null;
        _error = mod_json_error_null;
        _context = str;

        tok = mod_json_token_create(&_option);
        if (tok) {
            mod_json_value_t *jval;

            jval = mod_json_parse(tok, str);

            /* save information of token */
            _state = mod_json_token_state(tok);
            _error = mod_json_token_error(tok);
            _context = mod_json_token_context(tok);
            mod_json_token_destroy(tok);

            if (jval) {
                *out = *reinterpret_cast<JsonValue *>(&jval);
                mod_json_value_unset(jval);

                return out->isValid();
            }
        }
        return false;
    }

    //! Retrieve the error code of parser
    int error(void) const
    {
        return (int)_error;
    }

    //! Retrieve the state code of parser
    int state(void) const
    {
        return (int)_state;
    }

    //! Retrieve the context of parser
    const char *context(void) const
    {
        return _context;
    }

private:
    mod_json_option_t _option;
    mod_json_state_t _state;
    mod_json_error_t _error;
    mod_json_cchar_t *_context;
};

/*! JSON Dumper
 */
class JsonDumper
{
public:
    //! Constructor
    JsonDumper(void) : _str() {}

    //! Destructor
    ~JsonDumper(void) {}

    //! Dump a JSON value to string
    bool dump(const JsonValue &val)
    {
        mod_json_string_t *str;

        str = mod_json_dump(*((mod_json_value_t **)&val));
        _str = *reinterpret_cast<JsonString *>(&str);
        if (str) {
            mod_json_string_unset(str);
            return true;
        }
        return false;
    }

    //! Retrieve result of dumper
    JsonString &result(void)
    {
        return _str;
    }

    //! Retrieve result of dumper
    const JsonString &result(void) const
    {
        return _str;
    }

private:
    JsonString _str;
};

} // namespace mercury

//! Equality
static inline bool operator==(const mercury::JsonString &lhs, const char *rhs)
{
    const char *self = lhs.c_str();
    if (self == rhs) {
        return true;
    }

    if (self && rhs) {
        return (std::strcmp(self, rhs) == 0);
    }
    return false;
}

//! Equality
static inline bool operator==(const char *lhs, const mercury::JsonString &rhs)
{
    return (rhs == lhs);
}

//! Equality
static inline bool operator==(const mercury::JsonString &lhs,
                              const std::string &rhs)
{
    std::size_t ls = lhs.size();
    std::size_t rs = rhs.size();
    if (ls == 0 && rs == 0) {
        return true;
    }

    if (ls == rs) {
        const char *ld = lhs.data();
        const char *rd = rhs.data();

        if (ld && rd) {
            return (std::memcmp(ld, rd, ls) == 0);
        }
    }
    return false;
}

//! Equality
static inline bool operator==(const std::string &lhs,
                              const mercury::JsonString &rhs)
{
    return (rhs == lhs);
}

//! Equality
static inline bool operator==(const mercury::JsonString &lhs,
                              const mercury::JsonValue &rhs)
{
    return (rhs.isString() ? lhs == rhs.asString() : false);
}

//! Equality
static inline bool operator==(const mercury::JsonValue &lhs,
                              const mercury::JsonString &rhs)
{
    return (lhs.isString() ? lhs.asString() == rhs : false);
}

//! Equality
static inline bool operator==(const mercury::JsonArray &lhs,
                              const mercury::JsonValue &rhs)
{
    return (rhs.isArray() ? lhs == rhs.asArray() : false);
}

//! Equality
static inline bool operator==(const mercury::JsonValue &lhs,
                              const mercury::JsonArray &rhs)
{
    return (lhs.isArray() ? lhs.asArray() == rhs : false);
}

//! Equality
static inline bool operator==(const mercury::JsonObject &lhs,
                              const mercury::JsonValue &rhs)
{
    return (rhs.isObject() ? lhs == rhs.asObject() : false);
}

//! Equality
static inline bool operator==(const mercury::JsonValue &lhs,
                              const mercury::JsonObject &rhs)
{
    return (lhs.isObject() ? lhs.asObject() == rhs : false);
}

//! Equality
static inline bool operator==(const mercury::JsonValue &lhs, const bool &rhs)
{
    return (lhs.isBoolean() ? lhs.asBool() == rhs : false);
}

//! Equality
static inline bool operator==(const bool &lhs, const mercury::JsonValue &rhs)
{
    return (rhs.isBoolean() ? lhs == rhs.asBool() : false);
}

//! Equality
static inline bool operator==(const mercury::JsonValue &lhs, const char &rhs)
{
    return (lhs.isInteger() ? lhs.asInteger() == rhs : false);
}

//! Equality
static inline bool operator==(const char &lhs, const mercury::JsonValue &rhs)
{
    return (rhs.isInteger() ? lhs == rhs.asInteger() : false);
}

//! Equality
static inline bool operator==(const mercury::JsonValue &lhs, const short &rhs)
{
    return (lhs.isInteger() ? lhs.asInteger() == rhs : false);
}

//! Equality
static inline bool operator==(const short &lhs, const mercury::JsonValue &rhs)
{
    return (rhs.isInteger() ? lhs == rhs.asInteger() : false);
}

//! Equality
static inline bool operator==(const mercury::JsonValue &lhs, const int &rhs)
{
    return (lhs.isInteger() ? lhs.asInteger() == rhs : false);
}

//! Equality
static inline bool operator==(const int &lhs, const mercury::JsonValue &rhs)
{
    return (rhs.isInteger() ? lhs == rhs.asInteger() : false);
}

//! Equality
static inline bool operator==(const mercury::JsonValue &lhs, const long &rhs)
{
    return (lhs.isInteger() ? lhs.asInteger() == rhs : false);
}

//! Equality
static inline bool operator==(const long &lhs, const mercury::JsonValue &rhs)
{
    return (rhs.isInteger() ? lhs == rhs.asInteger() : false);
}

//! Equality
static inline bool operator==(const mercury::JsonValue &lhs,
                              const long long &rhs)
{
    return (lhs.isInteger() ? lhs.asInteger() == rhs : false);
}

//! Equality
static inline bool operator==(const long long &lhs,
                              const mercury::JsonValue &rhs)
{
    return (rhs.isInteger() ? lhs == rhs.asInteger() : false);
}

//! Equality
static inline bool operator==(const mercury::JsonValue &lhs, const float &rhs)
{
    if (lhs.isFloat()) {
        double diff = lhs.asFloat() - rhs;
        return ((diff < DBL_EPSILON) && (diff > -DBL_EPSILON));
    }
    return false;
}

//! Equality
static inline bool operator==(const float &lhs, const mercury::JsonValue &rhs)
{
    if (rhs.isFloat()) {
        double diff = rhs.asFloat() - lhs;
        return ((diff < DBL_EPSILON) && (diff > -DBL_EPSILON));
    }
    return false;
}

//! Equality
static inline bool operator==(const mercury::JsonValue &lhs, const double &rhs)
{
    if (lhs.isFloat()) {
        double diff = lhs.asFloat() - rhs;
        return ((diff < DBL_EPSILON) && (diff > -DBL_EPSILON));
    }
    return false;
}

//! Equality
static inline bool operator==(const double &lhs, const mercury::JsonValue &rhs)
{
    if (rhs.isFloat()) {
        double diff = rhs.asFloat() - lhs;
        return ((diff < DBL_EPSILON) && (diff > -DBL_EPSILON));
    }
    return false;
}

//! Equality
static inline bool operator==(const mercury::JsonValue &lhs,
                              const unsigned char &rhs)
{
    return (lhs.isInteger() ? lhs.asInteger() == rhs : false);
}

//! Equality
static inline bool operator==(const unsigned char &lhs,
                              const mercury::JsonValue &rhs)
{
    return (rhs.isInteger() ? lhs == rhs.asInteger() : false);
}

//! Equality
static inline bool operator==(const mercury::JsonValue &lhs,
                              const unsigned short &rhs)
{
    return (lhs.isInteger() ? lhs.asInteger() == rhs : false);
}

//! Equality
static inline bool operator==(const unsigned short &lhs,
                              const mercury::JsonValue &rhs)
{
    return (rhs.isInteger() ? lhs == rhs.asInteger() : false);
}

//! Equality
static inline bool operator==(const mercury::JsonValue &lhs,
                              const unsigned int &rhs)
{
    return (lhs.isInteger() ? lhs.asInteger() == rhs : false);
}

//! Equality
static inline bool operator==(const unsigned int &lhs,
                              const mercury::JsonValue &rhs)
{
    return (rhs.isInteger() ? lhs == rhs.asInteger() : false);
}

//! Equality
static inline bool operator==(const mercury::JsonValue &lhs,
                              const unsigned long &rhs)
{
    return (lhs.isInteger()
                ? lhs.asInteger() == mercury::JsonValue::integer_type(rhs)
                : false);
}

//! Equality
static inline bool operator==(const unsigned long &lhs,
                              const mercury::JsonValue &rhs)
{
    return (rhs.isInteger()
                ? mercury::JsonValue::integer_type(lhs) == rhs.asInteger()
                : false);
}

//! Equality
static inline bool operator==(const mercury::JsonValue &lhs,
                              const unsigned long long &rhs)
{
    return (lhs.isInteger()
                ? lhs.asInteger() == mercury::JsonValue::integer_type(rhs)
                : false);
}

//! Equality
static inline bool operator==(const unsigned long long &lhs,
                              const mercury::JsonValue &rhs)
{
    return (rhs.isInteger()
                ? mercury::JsonValue::integer_type(lhs) == rhs.asInteger()
                : false);
}

//! Equality
static inline bool operator==(const mercury::JsonValue &lhs, const char *rhs)
{
    return (lhs.isString() ? lhs.asString() == rhs : false);
}

//! Equality
static inline bool operator==(const char *lhs, const mercury::JsonValue &rhs)
{
    return (rhs.isString() ? lhs == rhs.asString() : false);
}

//! Equality
static inline bool operator==(const mercury::JsonValue &lhs,
                              const std::string &rhs)
{
    return (lhs.isString() ? lhs.asString() == rhs : false);
}

//! Equality
static inline bool operator==(const std::string &lhs,
                              const mercury::JsonValue &rhs)
{
    return (rhs.isString() ? lhs == rhs.asString() : false);
}

//! No equality
static inline bool operator!=(const mercury::JsonString &lhs, const char *rhs)
{
    return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const char *lhs, const mercury::JsonString &rhs)
{
    return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const mercury::JsonString &lhs,
                              const std::string &rhs)
{
    return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const std::string &lhs,
                              const mercury::JsonString &rhs)
{
    return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const mercury::JsonString &lhs,
                              const mercury::JsonValue &rhs)
{
    return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const mercury::JsonValue &lhs,
                              const mercury::JsonString &rhs)
{
    return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const mercury::JsonArray &lhs,
                              const mercury::JsonValue &rhs)
{
    return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const mercury::JsonValue &lhs,
                              const mercury::JsonArray &rhs)
{
    return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const mercury::JsonObject &lhs,
                              const mercury::JsonValue &rhs)
{
    return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const mercury::JsonValue &lhs,
                              const mercury::JsonObject &rhs)
{
    return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const mercury::JsonValue &lhs, const bool &rhs)
{
    return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const bool &lhs, const mercury::JsonValue &rhs)
{
    return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const mercury::JsonValue &lhs, const char &rhs)
{
    return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const char &lhs, const mercury::JsonValue &rhs)
{
    return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const mercury::JsonValue &lhs, const short &rhs)
{
    return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const short &lhs, const mercury::JsonValue &rhs)
{
    return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const mercury::JsonValue &lhs, const int &rhs)
{
    return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const int &lhs, const mercury::JsonValue &rhs)
{
    return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const mercury::JsonValue &lhs, const long &rhs)
{
    return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const long &lhs, const mercury::JsonValue &rhs)
{
    return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const mercury::JsonValue &lhs,
                              const long long &rhs)
{
    return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const long long &lhs,
                              const mercury::JsonValue &rhs)
{
    return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const mercury::JsonValue &lhs, const float &rhs)
{
    return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const float &lhs, const mercury::JsonValue &rhs)
{
    return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const mercury::JsonValue &lhs, const double &rhs)
{
    return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const double &lhs, const mercury::JsonValue &rhs)
{
    return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const mercury::JsonValue &lhs,
                              const unsigned char &rhs)
{
    return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const unsigned char &lhs,
                              const mercury::JsonValue &rhs)
{
    return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const mercury::JsonValue &lhs,
                              const unsigned short &rhs)
{
    return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const unsigned short &lhs,
                              const mercury::JsonValue &rhs)
{
    return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const mercury::JsonValue &lhs,
                              const unsigned int &rhs)
{
    return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const unsigned int &lhs,
                              const mercury::JsonValue &rhs)
{
    return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const mercury::JsonValue &lhs,
                              const unsigned long &rhs)
{
    return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const unsigned long &lhs,
                              const mercury::JsonValue &rhs)
{
    return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const mercury::JsonValue &lhs,
                              const unsigned long long &rhs)
{
    return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const unsigned long long &lhs,
                              const mercury::JsonValue &rhs)
{
    return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const mercury::JsonValue &lhs, const char *rhs)
{
    return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const char *lhs, const mercury::JsonValue &rhs)
{
    return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const mercury::JsonValue &lhs,
                              const std::string &rhs)
{
    return !(lhs == rhs);
}

//! No equality
static inline bool operator!=(const std::string &lhs,
                              const mercury::JsonValue &rhs)
{
    return !(lhs == rhs);
}

#endif // __MERCURY_UTILITY_JSON_MOD_JSON_PLUS_H__
