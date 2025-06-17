#pragma once

#include <string>
#include <sys/types.h>
#include <vector>
#include <memory>
#include <pthread.h>
#include <utility>
#include <assert.h>
#include <iostream>
#include <sys/stat.h>
#include <stdlib.h>
#include <malloc.h>
#include <errno.h>
#include <unistd.h>
#include <cmath>
#include "src/core/common/common.h"

MERCURY_NAMESPACE_BEGIN(core);

struct PackAttr
{
    docid_t docId;
};

struct BigBlock
{
    //if block is not full, next used as number of docs in block
    //if block is full, next used as offset to next block
    const static auto DOCS_PER_BLOCK = 1022;
    off_t next;
    PackAttr attr[DOCS_PER_BLOCK];
};

struct SmallBlock
{
    //if block is not full, next used as number of docs in block
    //if block is full, next used as offset to next block
    const static auto DOCS_PER_BLOCK = 30;
    off_t next;
    PackAttr attr[DOCS_PER_BLOCK];
};

struct IndexSlot
{
    size_t docCount;
    int64_t offset;
    int64_t lastOffset;
};

template <typename Block>
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
        off_t usedDocNum;
        char padding[16];
    };

    class PostingIterator
    {
    public:
        PostingIterator() 
            : _pIndexBase(nullptr), _pBlock(nullptr), _nextIdx(0), _docCount(0), _readCount(0) {}
        PostingIterator(char *pBase, Block *pBlock, size_t docCount)
            : _pIndexBase(pBase), _pBlock(pBlock), _nextIdx(0), _docCount(docCount), _readCount(0) {}

        size_t getDocNum() const {
            return _docCount;
        }

        // 不考虑加锁情况
        const docid_t& next() {
            //TODO remove it
            if (UNLIKELY(!_pBlock)) {
                return INVALID_DOC_ID;
            }
            if (_pBlock->next > Block::DOCS_PER_BLOCK) {
                if (_nextIdx < Block::DOCS_PER_BLOCK) {
                    _readCount++;
                    return _pBlock->attr[_nextIdx++].docId;
                } else {
                    _nextIdx = 0;
                    _pBlock = reinterpret_cast<Block *>(_pIndexBase + _pBlock->next);
                    _readCount++;
                    return _pBlock->attr[_nextIdx++].docId;
                }
            } else {
                if (_nextIdx < _pBlock->next) {
                    _readCount++;
                    return _pBlock->attr[_nextIdx++].docId;
                } else {
                    // last block
                    return INVALID_DOC_ID;
                }
            }
        };
        bool finish() const {

            if (_pBlock == nullptr
                || _readCount >= _docCount
                || (_pBlock && _pBlock->next <= Block::DOCS_PER_BLOCK
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
        size_t _docCount;
        size_t _readCount;
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

    int UpdateDocId(const std::vector<docid_t>&, size_t);

    PostingIterator search(int32_t coarseLabel) const;
    //PostingIterator linearSearch();



    bool dump(const std::string &file);
    void serialize(std::string &output) const;

    Block* getNextBlock(Block* curBlock) {
        assert(curBlock != nullptr);
        if (curBlock->next < Block::DOCS_PER_BLOCK) {
            return nullptr;
        }
        return reinterpret_cast<Block *>(_pBase + curBlock->next);
    }

    const void* GetBasePtr() const { return static_cast<const void*>(_pBase); }

    const IndexSlot *getIndexSlot() const {
        return _pIndexSlot;
    }
    const Header *getHeader() const {
        return _pHeader;
    }

    Header *getHeader() {
        return _pHeader;
    }

    size_t getSlotNum() const {
        return _pHeader->slotNum;
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
    docid_t getUsedDocNum() const {
        if (_pHeader) {
            return _pHeader->usedDocNum;
        }

        return INVALID_DOC_ID;
    }

    static size_t calcSize(size_t slotNum, size_t docNum);

    void PrintStats() const {
        std::cout << "slot num: " << _pHeader->slotNum << std::endl;
        for (auto i = 0; i < _pHeader->slotNum; ++i) {
            std::cout << "slot i:" << i << ", docCount:"  << _pIndexSlot[i].docCount << std::endl;
            // CoarseIndex::PostingIterator postingiterator = search(i);
            // while(!postingiterator.finish()){
            //     std::cout << postingiterator.next() << " ";
            // }
            // std::cout << std::endl;
        }
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
    //spin lock protecting _pHeader->usedDocNum
    pthread_spinlock_t _usedDocNumLock;
    //rwlock protecting Level2Node->offset
    pthread_rwlock_t _offsetLock[LOCK_NUMBER];
    //spin lock protecting Level2Node->lastOffset
    pthread_spinlock_t _lastOffsetLock[LOCK_NUMBER];
    //pthread_rwlock_t protecting Block->next in last Block
    pthread_rwlock_t _lastBlockLock[LOCK_NUMBER];
};

template<typename Block>
CoarseIndex<Block>::CoarseIndex()
{
    _pBase = nullptr;
    _pIndexSlot = nullptr;
    _pBlockPosting = nullptr;
    _pHeader = nullptr;
    pthread_spin_init(&_usedSizeLock, PTHREAD_PROCESS_PRIVATE);
    pthread_spin_init(&_usedDocNumLock, PTHREAD_PROCESS_PRIVATE);
    for (int i = 0; i < LOCK_NUMBER; ++i) {
        pthread_rwlock_init(_offsetLock + i, nullptr);
        pthread_spin_init(_lastOffsetLock + i, PTHREAD_PROCESS_PRIVATE);
        pthread_rwlock_init(_lastBlockLock + i, nullptr);
    }
}

template<typename Block>
CoarseIndex<Block>::~CoarseIndex()
{
    _pBase = nullptr;
    _pIndexSlot = nullptr;
    _pBlockPosting = nullptr;
    _pHeader = nullptr;

    pthread_spin_destroy(&_usedSizeLock);
    pthread_spin_destroy(&_usedDocNumLock);
    for (int i = 0; i < LOCK_NUMBER; ++i) {
        pthread_rwlock_destroy(_offsetLock + i);
        pthread_spin_destroy(_lastOffsetLock + i);
        pthread_rwlock_destroy(_lastBlockLock + i);
    }
}

template<typename Block>
bool CoarseIndex<Block>::create(void *pBase, size_t memorySize, int slotNum, int64_t maxDocSize)
{
    assert(pBase != nullptr);
    assert(maxDocSize > 0);
    assert(slotNum > 0);

    size_t capacity = calcSize(slotNum, maxDocSize);
    if (capacity > memorySize) {
        std::cerr << "Expected capacity() exceed memory size()" << std::endl;
        return false;
    }

    _pBase = (char*)pBase;
    memset(_pBase, 0, capacity);

    _pHeader = (Header*) pBase;
    _pHeader->docsPerBlock = Block::DOCS_PER_BLOCK;
    _pHeader->slotNum = slotNum;
    _pHeader->maxDocSize = maxDocSize;
    _pHeader->capacity = capacity;
    _pHeader->usedSize= sizeof(Header) + slotNum * sizeof(IndexSlot);
    _pHeader->usedDocNum = 0;
    memset(_pHeader->padding, 0, sizeof(_pHeader->padding));

    _pIndexSlot = (IndexSlot *)(reinterpret_cast<char*>(_pHeader) + sizeof(Header));
    _pBlockPosting = (Block *)(reinterpret_cast<char*>(_pIndexSlot) + sizeof(IndexSlot) * slotNum);
    return true;
}

template<typename Block>
bool CoarseIndex<Block>::load(void *pBase, size_t len)
{
    assert(pBase != nullptr);
    _pBase = (char *)pBase;
    _pHeader = (Header*) pBase;
    if (calcSize(_pHeader->slotNum, _pHeader->maxDocSize) != (size_t)_pHeader->capacity) {
        std::cerr << "expected index size is not equal to real file size" << std::endl;
        return false;
    }
    if ((size_t)_pHeader->capacity != len) {
        std::cerr << "file size in header is not equal to real file size" << std::endl;
        return false;
    }

    _pIndexSlot = (IndexSlot *)(reinterpret_cast<char*>(_pHeader) + sizeof(Header));
    _pBlockPosting = (Block *)(reinterpret_cast<char*>(_pIndexSlot) + sizeof(IndexSlot) * _pHeader->slotNum);
    // PrintStats();
    return true;
}

template<typename Block>
bool CoarseIndex<Block>::unload()
{
    //_pHeader->usedSize = sizeof(Header) + sizeof(IndexSlot) * _pHeader->slotNum;
    //memset(_pIndexSlot, 0xff, sizeof(IndexSlot) * _pHeader->slotNum);
    // TODO
    return true;
}

template<typename Block>
void CoarseIndex<Block>::reset()
{
    _pHeader->usedSize = sizeof(Header) + sizeof(IndexSlot) * _pHeader->slotNum;
    memset(_pBase + sizeof(Header), 0, _pHeader->capacity - sizeof(Header));
}

template<typename Block>
bool CoarseIndex<Block>::dump(const std::string &postingFile)
{
    FILE *fp = fopen(postingFile.c_str(), "wb");
    if (nullptr == fp) {
        std::cerr << "Fopen file [] with wb failed:" << std::endl;
        return false;
    }

    //header should be written in the end, to ensure integrity
    off_t headSize = sizeof(*_pHeader);
    int ret = fseek(fp, headSize, SEEK_SET);
    if (ret != 0) {
        std::cerr << "Seek file [] failed:" << std::endl;
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
            std::cerr << "Write to file [] failed:[], file size:, left size:" << std::endl;
            fclose(fp);
            return false;
        }

        curPos += writeSize;
        leftSize -= writeSize;
    }

    rewind(fp);
    off_t cnt = fwrite(_pHeader, headSize, 1, fp);
    if (cnt != 1) {
        std::cerr << "Write file head to file [] failed:[]" << std::endl;
        fclose(fp);
        return false;
    }

    fclose(fp);

    return true;
}

template<typename Block>
void CoarseIndex<Block>::serialize(std::string &output) const
{
    output = std::string(_pBase, _pHeader->capacity);
}

template<typename Block>
int CoarseIndex<Block>::UpdateDocId(const std::vector<docid_t>& docId_, size_t offset_) {
    for (int i = 0; i < _pHeader->slotNum; ++i) {
        auto&& iter = search(i);
        while (true) {
            const auto& docId = iter.next();
            if (docId == INVALID_DOC_ID) break;
            const_cast<docid_t&>(docId) = docId_.at(docId + offset_);
        }
    }
    return 0;
}

template<typename Block>
bool CoarseIndex<Block>::addDoc(int32_t coarseLabel, docid_t docId)
{
    Block *pCurBlock = nullptr;
    IndexSlot *slot = _pIndexSlot + coarseLabel;
    int32_t lockIdx = jumpConsistentHash(coarseLabel, LOCK_NUMBER);
    pthread_rwlock_rdlock(_offsetLock + lockIdx);
    if (UNLIKELY(slot->offset <= 0)) {
        pthread_rwlock_unlock(_offsetLock + lockIdx);
        pthread_rwlock_wrlock(_offsetLock + lockIdx);
        if (UNLIKELY(slot->offset > 0)) {
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
                std::cerr << "Memory exhausted(/), can't add more doc" << std::endl;
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
    while (pCurBlock->next > Block::DOCS_PER_BLOCK) {
        pCurBlock = reinterpret_cast<Block *>(_pBase + pCurBlock->next);
    }

    //current block is full, allocate a new block
    if (pCurBlock->next == Block::DOCS_PER_BLOCK) {
        off_t preUsedSize = 0;
        off_t blockSize = sizeof(Block);
        pthread_spin_lock(&_usedSizeLock);
        if (_pHeader->usedSize + blockSize > _pHeader->capacity) {
            pthread_spin_unlock(&_usedSizeLock);
            pthread_rwlock_unlock(_lastBlockLock + lockIdx);
            std::cerr << "Memory exhausted, can't add more doc to current chain" << std::endl;;
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

    pthread_spin_lock(&_usedDocNumLock);
    (_pHeader->usedDocNum)++;
    pthread_spin_unlock(&_usedDocNumLock);

    return true;
}

template<typename Block>
typename CoarseIndex<Block>::PostingIterator CoarseIndex<Block>::search(int32_t coarseLabel) const
{
    IndexSlot &slot = _pIndexSlot[coarseLabel];
    if (UNLIKELY(slot.offset <= 0)) {
        return PostingIterator();
    }
    Block *beginBlock = reinterpret_cast<Block*>(_pBase + slot.offset);
    return PostingIterator(_pBase, beginBlock, slot.docCount);
}

template<typename Block>
size_t CoarseIndex<Block>::calcSize(size_t slotNum, size_t maxDocSize)
{
    //the worst case:every chain only has one doc in last block
    // waste slotNum blocks to make sure capacity is big enough.
    size_t blockCnt = slotNum;
    if (maxDocSize > Block::DOCS_PER_BLOCK) {
        blockCnt = (size_t)ceil(1.0 * maxDocSize / Block::DOCS_PER_BLOCK) + slotNum;
    }
    return slotNum * sizeof(IndexSlot) + blockCnt * sizeof(Block) + sizeof(Header);
}

MERCURY_NAMESPACE_END(core);
