
/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     corase_index.h
 *   \author   yunheng@xiaohongshu.com
 *   \date     Jan 2019
 *   \version  1.0.0
 *   \brief    L1 Coarse Index
 */

#ifndef _MERCURY_INDEX_COARSE_INDEX_H_
#define _MERCURY_INDEX_COARSE_INDEX_H_

#include <string>
#include <sys/types.h>
#include <vector>
#include <memory>
#include <pthread.h>
#include <utility>
#include <assert.h>
#include "common/common_define.h"
#include <iostream>

#define DOCS_PER_BLOCK 1022

namespace mercury {

struct PackAttr
{
    docid_t docId;
};

struct Block
{
    //if block is not full, next used as number of docs in block
    //if block is full, next used as offset to next block
    off_t next;
    PackAttr attr[DOCS_PER_BLOCK];
};

struct IndexSlot
{
    size_t docCount;
    int64_t offset;
    int64_t lastOffset;
};


class CoarseIndex
{
public:
    typedef std::shared_ptr<CoarseIndex> Pointer;
    struct Header
    {
        off_t docsPerBlock;
        off_t slotNum;
        off_t maxDocSize;
        off_t capacity;
        off_t usedSize;
        char padding[24];
    };

    class PostingIterator
    {
    public:
        PostingIterator() 
            : _pIndexBase(nullptr), _pBlock(nullptr), _nextIdx(0) {}
        PostingIterator(char *pBase, Block *pBlock) 
            : _pIndexBase(pBase), _pBlock(pBlock), _nextIdx(0) {}

        // 不考虑加锁情况
        docid_t next() {
            //TODO remove it
            if (unlikely(!_pBlock)) {
                return INVALID_DOCID;
            }
            if (_pBlock->next > DOCS_PER_BLOCK) {
                if (_nextIdx < DOCS_PER_BLOCK) {
                    return _pBlock->attr[_nextIdx++].docId;
                } else {
                    _nextIdx = 0;
                    _pBlock = reinterpret_cast<Block *>(_pIndexBase + _pBlock->next);
                    return _pBlock->attr[_nextIdx++].docId;
                }
            } else {
                if (_nextIdx < _pBlock->next) {
                    return _pBlock->attr[_nextIdx++].docId;
                } else {
                    // last block
                    return INVALID_DOCID;
                }
            }
        };
        bool finish() const {

            if (_pBlock == nullptr || (_pBlock 
                && _pBlock->next <= DOCS_PER_BLOCK 
                && _nextIdx >= _pBlock->next)) {
                return true;
            } else {
                return false;
            }
        }

    private:
        char *_pIndexBase;
        Block *_pBlock;
        int _nextIdx;
    };

public:
    CoarseIndex();
    virtual ~CoarseIndex();

public:
    bool create(void *pBase, size_t memorySize, int slotNum, int64_t maxDocSize);
    bool load(void *pBase, size_t memorySize);
    bool unload();
    void reset();
    bool addDoc(int32_t coarseLabel, docid_t docId);

    PostingIterator search(int32_t coarseLabel) const;
    //PostingIterator linearSearch();

    bool dump(const std::string &file);
    void serialize(std::string &output) const;

    Block* getNextBlock(Block* curBlock) {
        assert(curBlock != nullptr);
        if (curBlock->next < DOCS_PER_BLOCK) {
            return nullptr;
        }
        return reinterpret_cast<Block *>(_pBase + curBlock->next);
    }

    const IndexSlot *getIndexSlot() const {
        return _pIndexSlot;
    }
    const Header *getHeader() const {
        return _pHeader;
    }
    const Block *getBlockPosting() const {
        return _pBlockPosting;
    }

public:
    static inline int32_t jumpConsistentHash(uint64_t key, int32_t numBuckets) 
    {
        int64_t b = -1;
        int64_t j = 0;
        while (j < numBuckets) {
            b = j;
            key = key * 2862933555777941757ULL + 1;
            j = (b + 1) * (static_cast<double>(1LL << 31) / static_cast<double>((key >> 33) + 1));
        }
        return b;
    }

public:
    static size_t calcSize(size_t slotNum, size_t docNum);

    void PrintStats() const {
        std::cout << "slot num: " << _pHeader->slotNum << std::endl;
        for (auto i = 0; i < _pHeader->slotNum; ++i) {
            std::cout << _pIndexSlot[i].docCount << ",";
        }
        std::cout << std::endl;
    }

private:
    static const int LOCK_NUMBER = 1024;

private:
    char *_pBase;
    IndexSlot *_pIndexSlot;
    Block *_pBlockPosting;
    Header *_pHeader;
    //spin lock protecting _pHeader->usedSize
    pthread_spinlock_t _usedSizeLock;
    //rwlock protecting Level2Node->offset
    pthread_rwlock_t _offsetLock[LOCK_NUMBER];
    //spin lock protecting Level2Node->lastOffset
    pthread_spinlock_t _lastOffsetLock[LOCK_NUMBER];
    //pthread_rwlock_t protecting Block->next in last Block
    pthread_rwlock_t _lastBlockLock[LOCK_NUMBER];
};

} // mercury

#endif //_MERCURY_INDEX_COARSE_INDEX_H_
