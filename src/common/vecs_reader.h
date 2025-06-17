#ifndef __MERCURY_VECS_READER_H__
#define __MERCURY_VECS_READER_H__

#include "common/common_define.h"
#include "framework/index_meta.h"
#include "framework/utility/mmap_file.h"
#include <iostream>

namespace mercury
{

class VecsReader
{
public:
    VecsReader(size_t vectorSize, const bool catEnabled = false)
        : _mmapFile(),
        _vectorSize(vectorSize),
        _numVecs(0),
        _vectorBase(nullptr),
        _keyBase(nullptr),
        _catBase(nullptr),
        _catEnables(catEnabled)
    {
    }

    bool load(const std::string &fname)
    {
        return load(fname.c_str());
    }

    bool load(const char *fname) 
    {
        if (!fname) { 
            std::cerr << "Load fname is nullptr" << std::endl;
            return false;
        }
        if (!_mmapFile.open(fname, true)) {
            std::cerr << "Open file error: " << fname << std::endl;
            return false;
        }
        size_t expectRecordSize = _vectorSize + sizeof(key_t);
        if (_catEnables) {
            expectRecordSize += sizeof(cat_t);
        }
        if (_mmapFile.region_size() % expectRecordSize != 0) {
            std::cerr << "File size is not match: " << _mmapFile.region_size()
                      << ", expectRecordSize: " << expectRecordSize << std::endl;
            return false;
        }

        // check
        _numVecs = _mmapFile.region_size() / expectRecordSize;
        _vectorBase = reinterpret_cast<const char *>(_mmapFile.region());
        _keyBase = reinterpret_cast<const key_t *>(_vectorBase + _numVecs * _vectorSize);
        if (_catEnables) {
            _catBase = reinterpret_cast<const cat_t *>(_keyBase + _numVecs);
        }
        return true;
    }

    size_t numVecs() const 
    {
        return _numVecs;
    }

    const void *vectorBase() const
    {
        return _vectorBase;
    }

    const uint64_t *keyBase() const
    {
        return _keyBase;
    }

    key_t getKey(size_t index) const
    {
        return _keyBase[index];
    }

    cat_t getCat(size_t index) const
    {
        if (_catEnables) {
            return _catBase[index];
        } else {
            return INVALID_CAT_ID;
        }
    }

    const void *getVector(size_t index) const
    {
        return _vectorBase + index * _vectorSize;
    }

private:
    mercury::MMapFile _mmapFile;
    size_t _vectorSize;
    size_t _numVecs;
    const char *_vectorBase;
    const uint64_t *_keyBase;
    const cat_t* _catBase;
    const bool _catEnables;
};

}; // namespace mercury

#endif //__MERCURY_VECS_READER_H__
