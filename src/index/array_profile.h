/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     array_profile.h
 *   \author   qiuming
 *   \date     Jan 2019
 *   \version  1.0.0
 *   \brief    interface of array profile
 */

#ifndef __MERCURY_ARRARY_PROFILE_H__
#define __MERCURY_ARRARY_PROFILE_H__

#include <stdint.h>
#include <assert.h>
#include <string>
#include <memory>
#include <memory.h>
#include <pthread.h>
#include "common/common_define.h"

namespace mercury {

class ArrayProfile
{
public:
    typedef std::shared_ptr<ArrayProfile> Pointer;
    struct Header
    {
    public:
        Header()
            : capacity(0)
            , usedDocNum(0)
            , maxDocNum(0)
            , infoSize(0)
        {};
        ~Header() {};
    public:
        int64_t capacity;
        int64_t usedDocNum;
        int64_t maxDocNum;
        int64_t infoSize;
        char padding[32];
    };

public:
    ArrayProfile();
    ~ArrayProfile();

private:
    ArrayProfile(const ArrayProfile &pqcodeProfile) = delete;
    ArrayProfile(ArrayProfile &&pqcodeProfile) = delete;
    ArrayProfile& operator=(const ArrayProfile &pqcodeProfile) = delete;

public:
    bool create(void *pBase, size_t memorySize, int64_t infoSize);
    bool load(void *pBase, size_t memorySize);
    bool unload();
    void reset();
    bool insert(docid_t docid, const void *info) 
    {
        assert(_base != nullptr);
        uint32_t idx = docid;
        if (likely(idx < _header->maxDocNum)) {
            memcpy(_infos + idx * _header->infoSize, info, _header->infoSize);
            _header->usedDocNum = docid + 1;
            return true;
        } else {
            return false;
        }
    }
    bool isFull()
    {
        //pthread_spin_lock(&_spinLock);
        bool res = _header->usedDocNum >= _header->maxDocNum;
        //pthread_spin_unlock(&_spinLock);
        return res;
    }

    // Retrieve info from profile
    const void *getInfo(docid_t docid) const 
    {
        int64_t idx = docid;
        if (likely(idx < _header->usedDocNum)) {
            return _infos + idx * _header->infoSize;
        }
        return nullptr;
    }

    const Header *getHeader() const
    {
        return _header;
    }

    size_t getDocNum() const
    {
        return _header->usedDocNum;
    }

    bool dump(const std::string &file);
    void serialize(std::string &output) const;

public:
    static size_t CalcSize(size_t docNum, size_t docSize);

private:
    char *_base;
    Header *_header;
    char *_infos;
    pthread_spinlock_t _spinLock;
};

}; // mercury

#endif // __MERCURY_ARRARY_PROFILE_H__
