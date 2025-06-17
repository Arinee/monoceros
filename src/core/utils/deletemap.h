#ifndef MERCURY_DELETEMAP_H_
#define MERCURY_DELETEMAP_H_

#include <stdint.h>
#include <string>
#include <memory.h>
#include <pthread.h>
#include "src/core/common/common_define.h"
#include "src/core/framework/index_framework.h"

MERCURY_NAMESPACE_BEGIN(core);

class DeleteMap
{
public:
    DeleteMap(uint64_t maxDocCnt);
    ~DeleteMap(void);

public:
    DeleteMap(const DeleteMap&) = delete;
    DeleteMap& operator=(const DeleteMap) = delete;

public:
    int mount(char *pBase, uint64_t size, bool bCreate = false);
    int unmount(void);

    uint64_t size(void) 
    {
        return _byteSize;
    }

    int setInvalid(docid_t docId)
    {
        docid_t quot = docId >> SLOT_SIZE_BIT_NUM;
        docid_t rem = docId & SLOT_SIZE_BIT_MASK;

        _pBitMap[quot] |= BITMAPOPMASK[rem];

        return 0;
    }

    int setAllInvalid(docid_t start, docid_t end)
    {
        docid_t startQuot = start >> SLOT_SIZE_BIT_NUM;
        docid_t endQuot = end >> SLOT_SIZE_BIT_NUM;
        for (docid_t i = startQuot; i <= endQuot; ++i) {
            _pBitMap[i] = INVALID_SLOT;
        }

        return 0;
    }

    int setAllInvalid(docid_t docId)
    {
        docid_t pos = docId >> SLOT_SIZE_BIT_NUM;
        _pBitMap[pos] = INVALID_SLOT;

        return 0;
    }

    int setBatchInvalid(docid_t start, docid_t end) 
    {
        docid_t startQuot = start >> SLOT_SIZE_BIT_NUM;
        docid_t startRem = start & SLOT_SIZE_BIT_MASK;

        docid_t endQuot = end >> SLOT_SIZE_BIT_NUM;
        docid_t endRem = end & SLOT_SIZE_BIT_MASK;
        //in the same slot
        if (startQuot == endQuot) {
            pthread_spin_lock(&_spinLock);
            for (docid_t i = startRem; i <= endRem; ++i) {
                _pBitMap[startQuot] |= BITMAPOPMASK[i];
            }
            pthread_spin_unlock(&_spinLock);
            return 0;
        }

        pthread_spin_lock(&_spinLock);
        //process start slot
        _pBitMap[startQuot] |= START_MASK[startRem];

        //process middle slots
        for (docid_t i = startQuot + 1; i < endQuot; ++i) {
            _pBitMap[i] = INVALID_SLOT;
        }

        //process end slot
        _pBitMap[endQuot] |= END_MASK[endRem];
        pthread_spin_unlock(&_spinLock);

        return 0;
    }

    int setValid(docid_t docId)
    {
        docid_t quot = docId >> SLOT_SIZE_BIT_NUM;
        docid_t rem = docId & SLOT_SIZE_BIT_MASK;

        _pBitMap[quot] &= ~BITMAPOPMASK[rem];

        return 0;
    }

    int setAllValid(docid_t docId)
    {
        docid_t pos = docId >> SLOT_SIZE_BIT_NUM;
        _pBitMap[pos] = VALID_SLOT;

        return 0;
    }

    int setAllValid(docid_t start, docid_t end)
    {
        docid_t startQuot = start >> SLOT_SIZE_BIT_NUM;
        docid_t endQuot = end >> SLOT_SIZE_BIT_NUM;
        for (docid_t i = startQuot; i <= endQuot; ++i) {
            _pBitMap[i] = VALID_SLOT;
        }

        return 0;
    }

    int setBatchValid(docid_t start, docid_t end) 
    {
        docid_t startQuot = start >> SLOT_SIZE_BIT_NUM;
        docid_t startRem = start & SLOT_SIZE_BIT_MASK;

        docid_t endQuot = end >> SLOT_SIZE_BIT_NUM;
        docid_t endRem = end & SLOT_SIZE_BIT_MASK;

        //in the same slot
        if (startQuot == endQuot) {
            pthread_spin_lock(&_spinLock);
            for (docid_t i = startRem; i <= endRem; ++i) {
                _pBitMap[startQuot] &= ~BITMAPOPMASK[i];
            }
            pthread_spin_unlock(&_spinLock);
            return 0;
        }

        pthread_spin_lock(&_spinLock);
        //process start slot
        _pBitMap[startQuot] &= ~START_MASK[startRem];

        //process middle slots
        for (docid_t i = startQuot + 1; i < endQuot; ++i) {
            _pBitMap[i] = VALID_SLOT;
        }

        //process end slot
        _pBitMap[endQuot] &= ~END_MASK[endRem];
        pthread_spin_unlock(&_spinLock);

        return 0;
    }

    bool isValid(docid_t docId)
    {
        docid_t quot = docId >> SLOT_SIZE_BIT_NUM;
        docid_t rem = docId & SLOT_SIZE_BIT_MASK;
        return (_pBitMap[quot] & BITMAPOPMASK[rem]) == 0;
    }

    bool isInvalid(docid_t docId)
    {
        docid_t quot = docId >> SLOT_SIZE_BIT_NUM;
        docid_t rem = docId & SLOT_SIZE_BIT_MASK;
        return (_pBitMap[quot] & BITMAPOPMASK[rem]) != 0;
    }

    int dump(const std::string &file);
    int dump(const std::string &file, 
             const mercury::core::IndexStorage::Pointer &stg);

public:
    static const std::string DELETE_MAP;
    static uint64_t needMemSize(uint64_t maxDocCnt);

public:
    static const uint64_t SLOT_SIZE = 64;
    static const uint64_t SLOT_SIZE_BIT_NUM = 6;
    static const uint64_t SLOT_SIZE_BIT_MASK = 0x3F;
    static const uint64_t BITMAPOPMASK[SLOT_SIZE];
    static const uint64_t START_MASK[SLOT_SIZE];
    static const uint64_t END_MASK[SLOT_SIZE];
    static const uint64_t VALID_SLOT = 0x0000000000000000;
    static const uint64_t INVALID_SLOT = 0xFFFFFFFFFFFFFFFF;
    
private:
    uint64_t _maxDocCnt;
    uint64_t _byteSize;
    uint64_t *_pBitMap;
    pthread_spinlock_t _spinLock;
};

MERCURY_NAMESPACE_END(core);
#endif //MERCURY_DELETEMAP_H_
