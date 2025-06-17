/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     mult_slot_index.h
 *   \author   yunheng@xiaohongshu.com
 *   \date     MAR 2019
 *   \version  1.0.0
 *   \brief    Mult Slot Index
 */

#ifndef _MERCURY_MULT_SLOT_INDEX_H_
#define _MERCURY_MULT_SLOT_INDEX_H_

#include <cmath>
#include <vector>
#include <memory>
#include <string>
#include <stdlib.h>
#include <malloc.h>
#include <errno.h>
#include <unistd.h>
#include <utility>
#include <assert.h>
#include <pthread.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "common/common_define.h"
#include "framework/index_logger.h"

#define DOCS_PER_BLOCK 1022

namespace mercury {

template<typename ValueType = docid_t>
class MultSlotIndex
{
public:
    typedef std::shared_ptr<MultSlotIndex<ValueType>> Pointer;

    struct Header
    {
        off_t docsPerBlock;
        off_t slotNum;
        off_t maxDocSize;
        off_t capacity;
        off_t usedSize;
        char padding[24];
    };

    struct PackAttr
    {
        ValueType docId;
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

    class PostingIterator
    {
    public:
        PostingIterator() 
            : _pIndexBase(nullptr), _pBlock(nullptr), _nextIdx(0) {}
        PostingIterator(char *pBase, Block *pBlock) 
            : _pIndexBase(pBase), _pBlock(pBlock), _nextIdx(0) {}

        // 不考虑加锁情况
        ValueType next() {
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
    MultSlotIndex(){
        _pBase = nullptr;
        _pIndexSlot = nullptr;
        _pBlockPosting = nullptr;
        _pHeader = nullptr;
        pthread_spin_init(&_usedSizeLock, PTHREAD_PROCESS_PRIVATE);
        for (int i = 0; i < LOCK_NUMBER; ++i) {
            pthread_rwlock_init(_offsetLock + i, nullptr);
            pthread_spin_init(_lastOffsetLock + i, PTHREAD_PROCESS_PRIVATE);
            pthread_rwlock_init(_lastBlockLock + i, nullptr);
        }   
    }
    virtual ~MultSlotIndex(){
        _pBase = nullptr;
        _pIndexSlot = nullptr;
        _pBlockPosting = nullptr;
        _pHeader = nullptr;

        pthread_spin_destroy(&_usedSizeLock);
        for (int i = 0; i < LOCK_NUMBER; ++i) {
            pthread_rwlock_destroy(_offsetLock + i);
            pthread_spin_destroy(_lastOffsetLock + i);
            pthread_rwlock_destroy(_lastBlockLock + i);
        }
    }

public:
    bool create(void *pBase, size_t memorySize, int slotNum, int64_t maxDocSize){
        assert(pBase != nullptr);
        assert(maxDocSize > 0);
        assert(slotNum > 0);

        size_t capacity = calcSize(slotNum, maxDocSize);
        if (capacity > memorySize) {
            LOG_ERROR("Expected capacity(%lu) exceed memory size(%lu)", capacity, memorySize);
            return false;
        }

        _pBase = (char*)pBase;
        memset(_pBase, 0, capacity);

        _pHeader = (Header*) pBase;
        _pHeader->docsPerBlock = DOCS_PER_BLOCK;
        _pHeader->slotNum = slotNum;
        _pHeader->maxDocSize = maxDocSize;
        _pHeader->capacity = capacity;
        _pHeader->usedSize= sizeof(Header) + slotNum * sizeof(IndexSlot);
        memset(_pHeader->padding, 0, sizeof(_pHeader->padding));

        _pIndexSlot = (IndexSlot *)(reinterpret_cast<char*>(_pHeader) + sizeof(Header));
        _pBlockPosting = (Block *)(reinterpret_cast<char*>(_pIndexSlot) + sizeof(IndexSlot) * slotNum);
        return true;
    }
    
    bool load(void *pBase, size_t memorySize){
        assert(pBase != nullptr);
        _pBase = (char *)pBase;
        _pHeader = (Header*) pBase;
        if (calcSize(_pHeader->slotNum, _pHeader->maxDocSize) != (size_t)_pHeader->capacity) {
            LOG_ERROR("expected index size is not equal to real file size");
            return false;
        }
        if ((size_t)_pHeader->capacity != memorySize) {
            LOG_ERROR("file size in header is not equal to real file size");
            return false;
        }

        _pIndexSlot = (IndexSlot *)(reinterpret_cast<char*>(_pHeader) + sizeof(Header));
        _pBlockPosting = (Block *)(reinterpret_cast<char*>(_pIndexSlot) + sizeof(IndexSlot) * _pHeader->slotNum);
        return true;
    }
    
    bool unload(){    
        //_pHeader->usedSize = sizeof(Header) + sizeof(IndexSlot) * _pHeader->slotNum;
        //memset(_pIndexSlot, 0xff, sizeof(IndexSlot) * _pHeader->slotNum);
        // TODO
        return true;
    }
    
    void reset(){
        _pHeader->usedSize = sizeof(Header) + sizeof(IndexSlot) * _pHeader->slotNum;
        memset(_pBase + sizeof(Header), 0, _pHeader->capacity - sizeof(Header));
    }

    bool addDoc(int32_t coarseLabel, ValueType docId){
        Block *pCurBlock = nullptr;
        IndexSlot *slot = _pIndexSlot + coarseLabel;
        int32_t lockIdx = jumpConsistentHash(coarseLabel, LOCK_NUMBER);
        pthread_rwlock_rdlock(_offsetLock + lockIdx);
        if (unlikely(slot->offset <= 0)) {
            pthread_rwlock_unlock(_offsetLock + lockIdx);
            pthread_rwlock_wrlock(_offsetLock + lockIdx);
            if (unlikely(slot->offset > 0)) {
                //other thread has malloced a new block and set it
                //offset never be changed from now on
                pthread_rwlock_unlock(_offsetLock + lockIdx);
                
                //get last block
                pthread_spin_lock(_lastOffsetLock + lockIdx);
                pCurBlock = reinterpret_cast<Block *>(_pBase + slot->lastOffset);
                pthread_spin_unlock(_lastOffsetLock + lockIdx);
            } else {
                off_t blockSize = sizeof(Block);
                off_t preUsedSize = -1;
                pthread_spin_lock(&_usedSizeLock);
                if (_pHeader->usedSize + blockSize > _pHeader->capacity) {
                    pthread_spin_unlock(&_usedSizeLock);
                    pthread_rwlock_unlock(_offsetLock + lockIdx);
                    LOG_ERROR("Memory exhausted(%ld/%ld), can't add more doc", _pHeader->usedSize, _pHeader->capacity);
                    return false;
                }

                preUsedSize = _pHeader->usedSize;
                _pHeader->usedSize += blockSize;
                pthread_spin_unlock(&_usedSizeLock);

                pthread_spin_lock(_lastOffsetLock + lockIdx);            
                slot->lastOffset = preUsedSize;
                pCurBlock = reinterpret_cast<Block *>(_pBase + slot->lastOffset);
                pCurBlock->next = 0;
                pthread_spin_unlock(_lastOffsetLock + lockIdx);                        
                //add to chain after block set Block.next
                slot->offset = preUsedSize;
                pthread_rwlock_unlock(_offsetLock + lockIdx);
            }

        } else {
            pthread_rwlock_unlock(_offsetLock + lockIdx);
            pthread_spin_lock(_lastOffsetLock + lockIdx);            
            pCurBlock = reinterpret_cast<Block *>(_pBase + slot->lastOffset);
            pthread_spin_unlock(_lastOffsetLock + lockIdx);            
        }

        pthread_rwlock_wrlock(_lastBlockLock + lockIdx);
        //other thread may add more doc after we get pCurBlock, we need walk again till last block
        while (pCurBlock->next > DOCS_PER_BLOCK) {
            pCurBlock = reinterpret_cast<Block *>(_pBase + pCurBlock->next);
        }

        //current block is full, allocate a new block
        if (pCurBlock->next == DOCS_PER_BLOCK) {
            off_t preUsedSize = 0;
            off_t blockSize = sizeof(Block);
            pthread_spin_lock(&_usedSizeLock);        
            if (_pHeader->usedSize + blockSize > _pHeader->capacity) {
                pthread_spin_unlock(&_usedSizeLock);
                pthread_rwlock_unlock(_lastBlockLock + lockIdx);            
                LOG_ERROR("Memory exhausted, can't add more doc to current chain");
                return false;
            }

            preUsedSize = _pHeader->usedSize;
            _pHeader->usedSize += blockSize;
            pthread_spin_unlock(&_usedSizeLock);

            Block *preBlock = pCurBlock;

            pthread_spin_lock(_lastOffsetLock + lockIdx);
            slot->lastOffset = preUsedSize;
            pCurBlock = reinterpret_cast<Block *>(_pBase + slot->lastOffset);
            pCurBlock->next = 0;
            pthread_spin_unlock(_lastOffsetLock + lockIdx);
            //add to chain after block set Block.next
            preBlock->next = preUsedSize;
        }

        PackAttr &attr = pCurBlock->attr[pCurBlock->next];
        attr.docId = docId;
        (pCurBlock->next)++;
        slot->docCount += 1;
        pthread_rwlock_unlock(_lastBlockLock + lockIdx);
        return true;
    }

    PostingIterator search(int32_t coarseLabel) const{
        IndexSlot &slot = _pIndexSlot[coarseLabel];
        if (unlikely(slot.offset <= 0)) {
            return PostingIterator();
        }
        Block *beginBlock = reinterpret_cast<Block*>(_pBase + slot.offset);
        return PostingIterator(_pBase, beginBlock);
    }

    bool dump(const std::string &postingFile){
        FILE *fp = fopen(postingFile.c_str(), "wb");
        if (nullptr == fp) {
            LOG_ERROR("Fopen file [%s] with wb failed:%s", postingFile.c_str(), strerror(errno));
            return false;
        }

        //header should be written in the end, to ensure integrity
        off_t headSize = sizeof(*_pHeader);
        int ret = fseek(fp, headSize, SEEK_SET);
        if (ret != 0) {
            LOG_ERROR("Seek file [%s] failed:%s", postingFile.c_str(), strerror(errno));
            fclose(fp);
            return false;
        }

        const off_t SIZE_ONE_TIME = 100 * 1024 * 1024;
        off_t leftSize = _pHeader->capacity - headSize;
        char *curPos = _pBase + headSize;
        while (leftSize > 0) {

            off_t curWriteSize = (leftSize < SIZE_ONE_TIME ? leftSize : SIZE_ONE_TIME);                
            off_t writeSize = fwrite(curPos, 1, curWriteSize, fp);
            if (writeSize != curWriteSize) {
                LOG_ERROR("Write to file [%s] failed:[%s], file size:%ld, left size:%ld", 
                          postingFile.c_str(), strerror(errno), _pHeader->capacity, leftSize);
                fclose(fp);
                return false;
            }

            curPos += writeSize;
            leftSize -= writeSize;
        }

        rewind(fp);
        off_t cnt = fwrite(_pHeader, headSize, 1, fp);
        if (cnt != 1) {
            LOG_ERROR("Write file head to file [%s] failed:[%s]", 
                      postingFile.c_str(), strerror(errno));
            fclose(fp);
            return false;
        }

        fclose(fp);

        return true;
    }

    void serialize(std::string &output) const{
        output = std::string(_pBase, _pHeader->capacity);
    }

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
    static size_t calcSize(size_t slotNum, size_t maxDocSize){
        //the worst case:every chain only has one doc in last block
        // waste slotNum blocks to make sure capacity is big enough.
        size_t blockCnt = slotNum;
        if (maxDocSize > DOCS_PER_BLOCK) {
            blockCnt = (size_t)ceil(1.0 * maxDocSize / DOCS_PER_BLOCK) + slotNum;
        }
        return slotNum * sizeof(IndexSlot) + blockCnt * sizeof(Block) + sizeof(Header);
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

#endif //_MERCURY_MULT_SLOT_INDEX_H_
