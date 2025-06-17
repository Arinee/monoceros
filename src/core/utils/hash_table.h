#ifndef MERCURY_HASH_TABLE_H_
#define MERCURY_HASH_TABLE_H_

#include <string>
#include <inttypes.h>
#include <pthread.h>
#include <cmath>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <memory>
#include "src/core/common/common_define.h"
#include "src/core/utils/dump_util.h"
#include "src/core/utils/local_file_writer.h"
#include "src/core/framework/index_framework.h"

MERCURY_NAMESPACE_BEGIN(core);
//bucket number is the frist prime which bigger than  maxItemCnt / BucketFactor
template <typename T, typename U, int BlockSize = 8, int BucketFactor = BlockSize>
class HashTable
{
public:
    typedef std::shared_ptr<HashTable> Pointer;
    HashTable(void);
    HashTable(const HashTable &hashTable) = delete;
    HashTable(HashTable &&hashTable) noexcept;
    ~HashTable(void);

public:
    HashTable &operator=(HashTable &&hashTable) noexcept;

public:
    struct HashTableHeader
    {
        uint64_t hashTableSize;
        uint32_t blockSize;
        uint32_t bucketCnt;
        uint64_t usedSize;
        uint64_t itemCnt;
        uint64_t maxItemCnt;
    };

    struct BlockHeader
    {
        uint64_t offset : 48;
        uint64_t size : 16;
    };

    struct Block
    {
        BlockHeader header;
        T keys[BlockSize];
        U values[BlockSize];
    };

public:
    static uint64_t needMemSize(uint64_t maxItemCnt);

public:
    //maxItemCnt must be set when bCreate is true
    int mount(char *pBase, uint64_t bufSize, 
              uint64_t maxItemCnt = 0UL, bool bCreate = false);
    void unmount(void);
    inline uint64_t size(void);
    inline uint64_t count(void);
    inline uint64_t needMemSize(void);
    int insert(const T &key, const U &value, bool bLimitCnt = false);
    int insertIfNotExist(const T &key, const U &value, bool bLimitCnt = false);
    bool remove(const T &key);
    bool find(const T &key, U &value);
    bool findQ(const T &key, U &value);
    void reset();
    int dump(const std::string &file);
    int dump(const std::string &file, const mercury::core::IndexStorage::Pointer &stg);
    inline char *getBase()
    {
        return _pBase;
    }

private:
    template <typename H>
    int doDump(H &handler);
    static uint32_t getBucketCnt(uint64_t maxItemCnt);
    void removeElement(Block *block, uint32_t deleteIdx);
    bool insertInNewBlock(uint64_t &newBlockOffset, const T &key, const U &value);
    void moveNode(Block *block, uint32_t deleteIdx);
    static uint32_t nextPrime(uint32_t baseSize);
    template <typename H>
    int writeBucket(H &handler, uint64_t offset, uint64_t value);
    template <typename H>
    int writeBlock(H &handler, uint64_t offset, Block *block);
    template <typename H>
    int processBlocks(H &handler, uint64_t &curOffset, uint64_t &blockOffset);
    template <typename H>
    int processBlock(H &handler, uint64_t &curOffset, Block *curBlock, Block &tmpBlock);

    inline uint64_t getHeaderSize() 
    {
        return sizeof(HashTableHeader);
    }

    inline uint64_t getBucketSize()
    {
        return sizeof(uint64_t) * _bucketCnt;
    }

    inline void increaseItemCnt()
    {
        pthread_spin_lock(&_itemCntLock);
        (*_pItemCnt)++;
        pthread_spin_unlock(&_itemCntLock);        
    }

    inline void decreaseItemCnt()
    {
        pthread_spin_lock(&_itemCntLock);
        (*_pItemCnt)--;
        pthread_spin_unlock(&_itemCntLock);        
    }
private:
    uint64_t _maxItemCnt;
    uint32_t _bucketCnt;
    HashTableHeader *_pHeader;
    uint64_t *_pItemCnt;
    uint64_t *_pBucket;
    char *_pBase;

    //spin lock protecting _pHeader->usedSize
    pthread_spinlock_t _usedSizeLock;
    pthread_spinlock_t _itemCntLock;
    static const uint32_t LOCK_NUMBER = 1024;//0x4000;
    static const uint32_t LOCK_MASK = 0x3FF;//0x3FFF;
    //spin lock  protecting last block
    pthread_rwlock_t _bucketLock[LOCK_NUMBER];    
};

template <typename T, typename U, int BlockSize, int BucketFactor>
HashTable<T, U, BlockSize, BucketFactor>::HashTable(void)
{
    _maxItemCnt = 0UL;
    _bucketCnt = 0U;
    _pHeader = nullptr;
    _pItemCnt = nullptr;
    _pBucket = nullptr;
    _pBase = nullptr;

    pthread_spin_init(&_usedSizeLock, PTHREAD_PROCESS_PRIVATE);
    pthread_spin_init(&_itemCntLock, PTHREAD_PROCESS_PRIVATE);
    for (uint32_t i = 0; i < LOCK_NUMBER; i++) {
        pthread_rwlock_init(_bucketLock + i, NULL);
    }
}

template <typename T, typename U, int BlockSize, int BucketFactor>
HashTable<T, U, BlockSize, BucketFactor>::HashTable(HashTable &&hashTable) noexcept
{
    _maxItemCnt = hashTable._maxItemCnt;
    _bucketCnt = hashTable._bucketCnt;
    _pHeader = hashTable._pHeader;
    _pItemCnt = hashTable._pItemCnt;
    _pBucket = hashTable._pBucket;
    _pBase = hashTable._pBase;

    pthread_spin_init(&_usedSizeLock, PTHREAD_PROCESS_PRIVATE);
    pthread_spin_init(&_itemCntLock, PTHREAD_PROCESS_PRIVATE);
    for (uint32_t i = 0; i < LOCK_NUMBER; i++) {
        pthread_rwlock_init(_bucketLock + i, NULL);
    }

    //reset hashTable
    hashTable.unmount();
    pthread_spin_destroy(&hashTable._usedSizeLock);
    pthread_spin_destroy(&hashTable._itemCntLock);
    for (uint32_t i = 0; i < LOCK_NUMBER; i++) {
        pthread_rwlock_destroy(hashTable._bucketLock + i);        
    }
}

template <typename T, typename U, int BlockSize, int BucketFactor>
HashTable<T, U, BlockSize, BucketFactor> &
HashTable<T, U, BlockSize, BucketFactor>::operator=(HashTable &&hashTable) noexcept
{
    if (this != &hashTable) {
        _maxItemCnt = hashTable._maxItemCnt;
        _bucketCnt = hashTable._bucketCnt;
        _pHeader = hashTable._pHeader;
        _pItemCnt = hashTable._pItemCnt;
        _pBucket = hashTable._pBucket;
        _pBase = hashTable._pBase;
        
        //reset hashTable
        hashTable.unmount();
        pthread_spin_destroy(&hashTable._usedSizeLock);
        pthread_spin_destroy(&hashTable._itemCntLock);
        for (uint32_t i = 0; i < LOCK_NUMBER; i++) {
            pthread_rwlock_destroy(hashTable._bucketLock + i);        
        }
    }

    return *this;
}

template <typename T, typename U, int BlockSize, int BucketFactor>
HashTable<T, U, BlockSize, BucketFactor>::~HashTable()
{
    this->unmount();

    pthread_spin_destroy(&_usedSizeLock);
    pthread_spin_destroy(&_itemCntLock);
    for (uint32_t i = 0; i < LOCK_NUMBER; i++) {
        pthread_rwlock_destroy(_bucketLock + i);        
    }
}

template <typename T, typename U, int BlockSize, int BucketFactor>
int HashTable<T, U, BlockSize, BucketFactor>::mount(char *pBase, uint64_t bufSize, 
        uint64_t maxItemCnt, bool bCreate)
{
    _pHeader = reinterpret_cast<HashTableHeader *>(pBase);
    _pBucket = reinterpret_cast<uint64_t *>(_pHeader + 1);
    _pBase = pBase;

    if (bCreate) {
        if (maxItemCnt == 0UL) {
            LOG_ERROR("Max item cnt must be greater than 0 when creat hash table!");
            return mercury::IndexError_InvalidArgument;
        }

        _maxItemCnt = maxItemCnt;
        _bucketCnt = getBucketCnt(_maxItemCnt);
        
        if (bufSize < needMemSize()) {
            LOG_ERROR("Mount size[%lu] less than need size [%lu]",
                      bufSize, needMemSize());
            return mercury::IndexError_InvalidLength;
        }

        _pHeader->hashTableSize = bufSize;
        _pHeader->blockSize = BlockSize;
        _pHeader->bucketCnt = _bucketCnt;
        _pHeader->usedSize = sizeof(HashTableHeader) + 
                             sizeof(uint64_t) * _pHeader->bucketCnt;
        _pHeader->itemCnt = 0;
        _pHeader->maxItemCnt = _maxItemCnt;

        _pItemCnt = &(_pHeader->itemCnt);
        memset(reinterpret_cast<void *>(_pBucket), static_cast<int>(INVALID_OFFSET), 
               sizeof(uint64_t) * _pHeader->bucketCnt);

        return 0;
    }

    //check size
    if (_pHeader->hashTableSize > bufSize) {
        LOG_ERROR("Mount size [%lu] less than hash table size [%lu]",
                  bufSize, _pHeader->hashTableSize);
        return mercury::IndexError_InvalidLength;
    }

    if (_pHeader->blockSize != BlockSize) {
        LOG_ERROR("Index block size [%u], doesn't match required in code [%d] ",
                  _pHeader->blockSize, BlockSize);
        return mercury::IndexError_Mismatch;
    }

    if (_pHeader->usedSize > _pHeader->hashTableSize) {
        LOG_ERROR("Used size [%lu] can't bigger than hash table size [%lu]",
                  _pHeader->usedSize, _pHeader->hashTableSize);
        return mercury::IndexError_InvalidValue;
    }

    if (_pHeader->maxItemCnt == 0UL) {
        LOG_ERROR("Max item cnt must be greater than 0 in index!");
        return mercury::IndexError_InvalidValue;
    }
    _maxItemCnt = _pHeader->maxItemCnt;
    _bucketCnt = _pHeader->bucketCnt;
    //maybe user extend hash table size
    if (bufSize > _pHeader->hashTableSize) {
        _pHeader->hashTableSize = bufSize;
    }
    _pItemCnt = &(_pHeader->itemCnt);

    return 0;
}

template <typename T, typename U, int BlockSize, int BucketFactor>
void HashTable<T, U, BlockSize, BucketFactor>::unmount(void)
{
    _pHeader = nullptr;
    _pItemCnt = nullptr;
    _pBucket = nullptr;
    _pBase = nullptr;

    return;
}

template <typename T, typename U, int BlockSize, int BucketFactor>
int HashTable<T, U, BlockSize, BucketFactor>::insert(const T &key, const U &value, bool bLimitCnt)
{
    // in limit count mode item count can't bigger than max itme count
    if (bLimitCnt && _pHeader->itemCnt >= _pHeader->maxItemCnt) {
        LOG_ERROR("Insert in new block failed");
        return mercury::IndexError_IndexFull;
    }

    uint32_t bucketIdx = key % _bucketCnt;
    uint32_t lockIdx = bucketIdx & LOCK_MASK;

    pthread_rwlock_wrlock(_bucketLock + lockIdx);
    uint64_t offset = _pBucket[bucketIdx];
    if (unlikely(offset == INVALID_OFFSET)) {
        uint64_t newBlockOffset = 0;
        bool bRet = insertInNewBlock(newBlockOffset, key, value);
        if (!bRet) {
            pthread_rwlock_unlock(_bucketLock + lockIdx);
            LOG_ERROR("Insert in new block failed");
            return mercury::IndexError_IndexFull;
        }

        //link to chain
        _pBucket[bucketIdx] = newBlockOffset; 
        increaseItemCnt();
    } else {
        Block *curBlock = reinterpret_cast<Block *>(_pBase + offset);
        //walk ahead to find a block where can insert or allocate a new block
        while (((curBlock->header).offset > 0) && ((curBlock->header).size == BlockSize)) {
            curBlock = reinterpret_cast<Block *>(_pBase + (curBlock->header).offset);
        }

        //need allocate new Block
        if ((curBlock->header).size == BlockSize) {
            uint64_t newBlockOffset = 0;
            bool bRet = insertInNewBlock(newBlockOffset, key, value);
            if (!bRet) {
                pthread_rwlock_unlock(_bucketLock + lockIdx);
                LOG_ERROR("Insert in new block failed");
                return mercury::IndexError_IndexFull;
            }

            increaseItemCnt();
            //link to chain
            curBlock->header.offset = newBlockOffset;
            pthread_rwlock_unlock(_bucketLock + lockIdx);

            return 0;
        }

        //else insert to current block
        (curBlock->keys)[(curBlock->header).size] = key;
        (curBlock->values)[(curBlock->header).size] = value;
        (curBlock->header).size++;
        this->increaseItemCnt();
    }
    pthread_rwlock_unlock(_bucketLock + lockIdx);

    return 0;
}

template <typename T, typename U, int BlockSize, int BucketFactor>
int HashTable<T, U, BlockSize, BucketFactor>::insertIfNotExist(const T &key, const U &value, bool bLimitCnt)
{
    // in limit count mode item count can't bigger than max itme count
    if (bLimitCnt && _pHeader->itemCnt >= _pHeader->maxItemCnt) {
        LOG_ERROR("Insert in new block failed");
        return mercury::IndexError_IndexFull;
    }

    uint32_t bucketIdx = key % _bucketCnt;
    uint32_t lockIdx = bucketIdx & LOCK_MASK;

    pthread_rwlock_wrlock(_bucketLock + lockIdx);
    uint64_t offset = _pBucket[bucketIdx];
    if (unlikely(offset == INVALID_OFFSET)) {
        uint64_t newBlockOffset = 0;
        bool bRet = insertInNewBlock(newBlockOffset, key, value);
        if (!bRet) {
            pthread_rwlock_unlock(_bucketLock + lockIdx);
            LOG_ERROR("Insert in new block failed");
            return mercury::IndexError_IndexFull;
        }

        //link to chain
        _pBucket[bucketIdx] = newBlockOffset;
        increaseItemCnt();
    } else {
        Block *insertBlock = nullptr;
        Block *curBlock = reinterpret_cast<Block *>(_pBase + offset);
        while ((curBlock->header).offset > 0) {
            //find a block can do insertion
            uint32_t number = (curBlock->header).size;
            if (number != BlockSize) {
                insertBlock = curBlock;
            }

            for (uint32_t i = 0; i < number; i++) {
                if (key == (curBlock->keys)[i]) {
                    pthread_rwlock_unlock(_bucketLock + lockIdx);
                    LOG_ERROR("Key already exists");
                    return mercury::IndexError_Exist;
                }
            }
            curBlock = reinterpret_cast<Block *>(_pBase + (curBlock->header).offset);
        }

        //see last block whether exists key
        uint32_t number = (curBlock->header).size;
        for (uint32_t i = 0; i < number; i++) {
            if (key == (curBlock->keys)[i]) {
                pthread_rwlock_unlock(_bucketLock + lockIdx);
                LOG_ERROR("Key already exists");
                return mercury::IndexError_Exist;
            }
        }

        //found a block to insert before last block
        if (insertBlock != nullptr) {
            (insertBlock->keys)[(insertBlock->header).size] = key;
            (insertBlock->values)[(insertBlock->header).size] = value;
            (insertBlock->header).size++;
            pthread_rwlock_unlock(_bucketLock + lockIdx);

            increaseItemCnt();

            return 0;
        }

        //whether the last block can do insertion
        if (number == BlockSize) {
            uint64_t newBlockOffset = 0;
            bool bRet = insertInNewBlock(newBlockOffset, key, value);
            if (!bRet) {
                pthread_rwlock_unlock(_bucketLock + lockIdx);
                LOG_ERROR("Insert in new block failed");
                return mercury::IndexError_NoMemory;
            }

            //link to chain
            curBlock->header.offset = newBlockOffset;
            pthread_rwlock_unlock(_bucketLock + lockIdx);
            increaseItemCnt();

            return 0;
        } else {
            (curBlock->keys)[number] = key;
            (curBlock->values)[number] = value;
            (curBlock->header).size++;

            increaseItemCnt();
        }
    }
    pthread_rwlock_unlock(_bucketLock + lockIdx);

    return 0;
}

template <typename T, typename U, int BlockSize, int BucketFactor>
bool HashTable<T, U, BlockSize, BucketFactor>::remove(const T &key)
{
    uint32_t bucketIdx = key % _bucketCnt;
    uint64_t offset = _pBucket[bucketIdx];
    if (offset == INVALID_OFFSET) {
        return false;
    }

    Block *curBlock = reinterpret_cast<Block *>(_pBase + offset);
    uint32_t lockIdx = bucketIdx & LOCK_MASK;
    pthread_rwlock_wrlock(_bucketLock + lockIdx);
    while ((curBlock->header).offset > 0) {
        uint32_t number = (curBlock->header).size;
        for (uint32_t i = 0; i < number; i++) {
            if (key == (curBlock->keys)[i]) {
                //move keys and values
                removeElement(curBlock, i);
                pthread_rwlock_unlock(_bucketLock + lockIdx);
                this->decreaseItemCnt();
                return true;
            }
        }
        curBlock = reinterpret_cast<Block *>(_pBase + (curBlock->header).offset);
    }

    //last block
    uint32_t number = (curBlock->header).size;
    for (uint32_t i = 0; i < number; i++) {
        if (key == (curBlock->keys)[i]) {
            removeElement(curBlock, i);
            pthread_rwlock_unlock(_bucketLock + lockIdx);
            this->decreaseItemCnt();
            return true;
        }
    }
    pthread_rwlock_unlock(_bucketLock + lockIdx);

    return false;
}

template <typename T, typename U, int BlockSize, int BucketFactor>
bool HashTable<T, U, BlockSize, BucketFactor>::find(const T &key, U &value)
{
    uint32_t bucketIdx = key % _bucketCnt;
    uint64_t offset = _pBucket[bucketIdx];
    if (offset == INVALID_OFFSET) {
        return false;
    }

    Block *curBlock = reinterpret_cast<Block *>(_pBase + offset);
    uint32_t lockIdx = bucketIdx & LOCK_MASK;
    pthread_rwlock_rdlock(_bucketLock + lockIdx);
    while ((curBlock->header).offset > 0) {
        uint32_t number = (curBlock->header).size;
        for (uint32_t i = 0; i < number; i++) {
            if (key == (curBlock->keys)[i]) {
                value = (curBlock->values)[i];
                pthread_rwlock_unlock(_bucketLock + lockIdx);
                return true;
            }
        }
        curBlock = reinterpret_cast<Block *>(_pBase + (curBlock->header).offset);
    }

    //last block
    uint32_t number = (curBlock->header).size;
    for (uint32_t i = 0; i < number; i++) {
        if (key == (curBlock->keys)[i]) {
            value = (curBlock->values)[i];
            pthread_rwlock_unlock(_bucketLock + lockIdx);
            return true;
        }
    }
    pthread_rwlock_unlock(_bucketLock + lockIdx);

    return false;
}

template <typename T, typename U, int BlockSize, int BucketFactor>
bool HashTable<T, U, BlockSize, BucketFactor>::findQ(const T &key, U &value)
{
    uint32_t bucketIdx = key % _bucketCnt;
    uint64_t offset = _pBucket[bucketIdx];
    if (offset == INVALID_OFFSET) {
        return false;
    }

    Block *curBlock = reinterpret_cast<Block *>(_pBase + offset);
    //uint32_t lockIdx = bucketIdx & LOCK_MASK;
    //pthread_rwlock_rdlock(_bucketLock + lockIdx);
    while ((curBlock->header).offset > 0) {
        uint32_t number = (curBlock->header).size;
        for (uint32_t i = 0; i < number; i++) {
            if (key == (curBlock->keys)[i]) {
                value = (curBlock->values)[i];
                //pthread_rwlock_unlock(_bucketLock + lockIdx);
                return true;
            }
        }
        curBlock = reinterpret_cast<Block *>(_pBase + (curBlock->header).offset);
    }

    //last block
    uint32_t number = (curBlock->header).size;
    for (uint32_t i = 0; i < number; i++) {
        if (key == (curBlock->keys)[i]) {
            value = (curBlock->values)[i];
            //pthread_rwlock_unlock(_bucketLock + lockIdx);
            return true;
        }
    }
    //pthread_rwlock_unlock(_bucketLock + lockIdx);

    return false;
}

template <typename T, typename U, int BlockSize, int BucketFactor>
inline void HashTable<T, U, BlockSize, BucketFactor>::reset(void)
{
    if (_pHeader == nullptr) {
        return;
    }

    _pHeader->usedSize = sizeof(HashTableHeader) +
                         sizeof(uint64_t) * _pHeader->bucketCnt;
    _pHeader->itemCnt = 0;

    memset(reinterpret_cast<void *>(_pBucket), static_cast<int>(INVALID_OFFSET),
           sizeof(uint64_t) * _pHeader->bucketCnt);
    return;
}

template <typename T, typename U, int BlockSize, int BucketFactor>
template <typename H>
int HashTable<T, U, BlockSize, BucketFactor>::doDump(H &handler)
{
    //write header
    size_t writeCnt = handler->write(&_pHeader, sizeof(HashTableHeader));
    if (writeCnt != sizeof(HashTableHeader)) {
        LOG_ERROR("Write HashTableHeader to file failed");
        return mercury::IndexError_WriteFile;
    }
    
    //scan every bucket, write its blocks
    uint64_t curOffset = getHeaderSize() + getBucketSize();
    uint64_t bucketOffset = getHeaderSize();
    for (uint32_t i = 0; i < _bucketCnt; i++) {
        //if current bucket is empty
        uint64_t blockOffset = _pBucket[i];
        if (blockOffset == INVALID_OFFSET) {
            int iRet = writeBucket<H>(handler, bucketOffset, blockOffset);
            if (iRet != 0) {
                LOG_ERROR("Write bucket failed");
                return iRet;
            }
            bucketOffset += sizeof(blockOffset);
            continue;
        }


        //write all blocks belong to current bucket
        uint64_t preOffset = curOffset;
        int iRet = processBlocks<H>(handler, curOffset, blockOffset);
        if (iRet != 0) {
            LOG_ERROR("Process blocks failed");
            return iRet;
        }

        //write block offset to corresponding bucket
        blockOffset = INVALID_OFFSET;
        //if write at least one block
        if (curOffset != preOffset) {
            blockOffset = preOffset;
        }

        iRet = writeBucket<H>(handler, bucketOffset, blockOffset);
        if (iRet != 0) {
            LOG_ERROR("Write bucket failed");
            return iRet;
        }
        bucketOffset += sizeof(blockOffset);            

    }
    //update header
    HashTableHeader header(*_pHeader);
    header.usedSize = curOffset;
    writeCnt = handler->write(0UL, &header, sizeof(header));
    if (writeCnt != sizeof(header)) {
        LOG_ERROR("Write file failed while updating header");
        return mercury::IndexError_WriteFile;
    }
    
    return 0;
}

template <typename T, typename U, int BlockSize, int BucketFactor>
int HashTable<T, U, BlockSize, BucketFactor>::dump(const std::string &file)
{
    LocalFileWriter::Pointer handler(new (std::nothrow) LocalFileWriter(file));
    if (!handler) {
        LOG_ERROR("New LocalFileWriter failed when dumping hash table");
        return mercury::IndexError_NoMemory;
    }

    bool bRet = handler->init();
    if (!bRet) {
        LOG_ERROR("Init FileWriter failed");
        return mercury::IndexError_OpenFile;
    }

    //do dump
    int iRet = doDump<LocalFileWriter::Pointer>(handler);
    if (iRet != 0) {
        LOG_ERROR("Error happens in doDump");
        return iRet;
    }

    iRet = handler->truncate(_pHeader->hashTableSize);
    if (iRet != 0) {
        LOG_ERROR("Ftruncate file failed");
        return mercury::IndexError_TruncateFile;
    }

    return 0;
}

template <typename T, typename U, int BlockSize, int BucketFactor>
int HashTable<T, U, BlockSize, BucketFactor>::dump(const std::string &file, 
                                                   const mercury::core::IndexStorage::Pointer &stg)
{
    if (!stg) {
        LOG_ERROR("Storage is nullptr");
        return mercury::IndexError_InvalidArgument;
    }

    if (!(stg->hasRandWrite())) {
        return DumpUtil::dump(_pBase, _pHeader->hashTableSize, file, stg);
    }

    //create handler
    auto handler = stg->create(file, _pHeader->hashTableSize);
    if (!handler) {
        LOG_ERROR("Storage create handler  failed");
        return mercury::IndexError_CreateStorageHandler;
    }

    int iRet = doDump<decltype(handler)>(handler);
    if (iRet != 0) {
        LOG_ERROR("Error happens in doDump");
        return iRet;
    }

    return 0;
}

template <typename T, typename U, int BlockSize, int BucketFactor>
inline uint64_t HashTable<T, U, BlockSize, BucketFactor>::size(void)
{
    return (_pHeader == nullptr) ? 0UL : _pHeader->hashTableSize;
}

template <typename T, typename U, int BlockSize, int BucketFactor>
inline uint64_t HashTable<T, U, BlockSize, BucketFactor>::count(void)
{
    return (_pItemCnt == nullptr) ? 0UL : (*_pItemCnt);
}

template <typename T, typename U, int BlockSize, int BucketFactor>
inline uint64_t HashTable<T, U, BlockSize, BucketFactor>::needMemSize(void)
{
    uint64_t blockCnt = 0UL;
    //the worst case:every last block only has one vector
    uint64_t elementsInLastBlock = _bucketCnt;
    if (_maxItemCnt < _bucketCnt) {
        blockCnt = _maxItemCnt;
    } else if (_maxItemCnt < elementsInLastBlock) {
        blockCnt = _bucketCnt;
    } else {
        blockCnt = (_maxItemCnt - elementsInLastBlock) / BlockSize + _bucketCnt;
    }

    return sizeof(HashTableHeader) + sizeof(uint64_t) * _bucketCnt +
        sizeof(Block) * blockCnt;
}

template <typename T, typename U, int BlockSize, int BucketFactor>
inline uint64_t HashTable<T, U, BlockSize, BucketFactor>::needMemSize(uint64_t maxItemCnt)
{
    uint64_t blockCnt = 0UL;
    uint32_t bucketCnt = getBucketCnt(maxItemCnt);
    //the worst case:every last block only has one vector
    uint64_t elementsInLastBlock = bucketCnt;
    if (maxItemCnt < bucketCnt) {
        blockCnt = maxItemCnt;
    } else if (maxItemCnt < elementsInLastBlock) {
        blockCnt = bucketCnt;
    } else {
        blockCnt = (maxItemCnt - elementsInLastBlock) / BlockSize + bucketCnt;
    }

    return sizeof(HashTableHeader) + sizeof(uint64_t) * bucketCnt +
        sizeof(Block) * blockCnt;
}

template <typename T, typename U, int BlockSize, int BucketFactor>
inline uint32_t HashTable<T, U, BlockSize, BucketFactor>::getBucketCnt(uint64_t maxItemCnt)
{
    uint32_t baseSize =  (maxItemCnt + BucketFactor - 1) / BucketFactor;
    return nextPrime(baseSize);
}

template <typename T, typename U, int BlockSize, int BucketFactor>
inline void HashTable<T, U, BlockSize, BucketFactor>::removeElement(Block *block, 
        uint32_t deleteIdx)
{
    memcpy(block->keys + deleteIdx, block->keys + deleteIdx + 1, 
           sizeof(T) * ((block->header).size - deleteIdx - 1));
    memcpy(block->values + deleteIdx, block->values + deleteIdx + 1, 
           sizeof(U) * ((block->header).size - deleteIdx - 1));
    (block->header).size--;
    return;
}

template <typename T, typename U, int BlockSize, int BucketFactor>
inline bool HashTable<T, U, BlockSize, BucketFactor>::insertInNewBlock(uint64_t &newBlockOffset, 
        const T &key, const U &value)
{
    pthread_spin_lock(&_usedSizeLock);
    if (_pHeader->usedSize + sizeof(Block) > _pHeader->hashTableSize) {
        pthread_spin_unlock(&_usedSizeLock);
        LOG_ERROR("Memory exhausted, can't add more element");
        return false;
    }
    newBlockOffset = _pHeader->usedSize;
    _pHeader->usedSize += sizeof(Block);
    pthread_spin_unlock(&_usedSizeLock);

    Block *newBlock = reinterpret_cast<Block *>(_pBase + newBlockOffset);
    (newBlock->header).offset = 0;
    (newBlock->header).size = 1;
    (newBlock->keys)[0] = key;
    (newBlock->values)[0] = value;

    return true;
}

template <typename T, typename U, int BlockSize, int BucketFactor>
inline uint32_t HashTable<T, U, BlockSize, BucketFactor>::nextPrime(uint32_t baseSize)
{
    if (baseSize <= 3U) {
        return 3U;
    }

    while (true) {
        uint32_t max = static_cast<uint32_t>(sqrt(baseSize));
        bool isPrime = true;
        for (uint32_t j = 2; j <= max; j++) {
            if ((baseSize % j) == 0) {
                isPrime = false;
                break;
            }
        }

        if (isPrime) {
            break;
        } 
            
        baseSize++;
    }

    return baseSize;
}

template <typename T, typename U, int BlockSize, int BucketFactor>
template <typename H>
inline int HashTable<T, U, BlockSize, BucketFactor>::writeBucket(H &handler, 
        uint64_t offset, uint64_t value)
{
    size_t writeCnt = handler->write(offset, &value, sizeof(value));
    if (writeCnt != sizeof(value)) {
        LOG_ERROR("Write file failed");
        return mercury::IndexError_WriteFile;
    }    

    return 0;
}

template <typename T, typename U, int BlockSize, int BucketFactor>
template <typename H>
inline int HashTable<T, U, BlockSize, BucketFactor>::writeBlock(H &handler, 
        uint64_t offset, Block *block)
{
    size_t writeCnt = handler->write(offset, block, sizeof(Block));
    if (writeCnt != sizeof(Block)) {
        LOG_ERROR("Write file failed");
        return mercury::IndexError_WriteFile;
    }

    return 0;
}

template <typename T, typename U, int BlockSize, int BucketFactor>
template <typename H>
int HashTable<T, U, BlockSize, BucketFactor>::processBlocks(H &handler, 
        uint64_t &curOffset, uint64_t &blockOffset)
{
    Block tmpBlock;
    tmpBlock.header.offset = curOffset + sizeof(tmpBlock);
    tmpBlock.header.size = 0;
    Block *curBlock = reinterpret_cast<Block *>(_pBase + blockOffset);
    while ((curBlock->header).offset != 0UL) {
        int ret = processBlock<H>(handler, curOffset, curBlock, tmpBlock);
        if (ret != 0) {
            LOG_ERROR("Process block failed");
            return ret;
        }
        curBlock = reinterpret_cast<Block *>(_pBase + (curBlock->header).offset);
    }

    //process last block
    int ret = processBlock<H>(handler, curOffset, curBlock, tmpBlock);
    if (ret != 0) {
        LOG_ERROR("Process block failed");
        return ret;
    }

    //write last block buffer
    if (tmpBlock.header.size != 0) {
        tmpBlock.header.offset = 0;
        ret = writeBlock<H>(handler, curOffset, &tmpBlock);
        if (ret != 0) {
            LOG_ERROR("Write block failed");
            return ret;
        }
        curOffset += sizeof(tmpBlock);
    }
    
    return 0;
}

template <typename T, typename U, int BlockSize, int BucketFactor>
template <typename H>
inline int HashTable<T, U, BlockSize, BucketFactor>::processBlock(H &handler, 
        uint64_t &curOffset, Block *curBlock, Block &tmpBlock)
{
    uint32_t number = (curBlock->header).size;
    for (uint32_t i = 0; i < number; i++) {
        //block buffer is full, dump to file
        if (tmpBlock.header.size == BlockSize) {
            int ret = writeBlock<H>(handler, curOffset, &tmpBlock);
            if (ret != 0) {
                LOG_ERROR("Write block failed");
                return ret;
            }
                    
            curOffset += sizeof(tmpBlock);
            tmpBlock.header.offset = curOffset + sizeof(tmpBlock);
            tmpBlock.header.size = 0;        
        }
        //write to block buffer
        tmpBlock.keys[tmpBlock.header.size] = (curBlock->keys)[i];
        tmpBlock.values[tmpBlock.header.size] = (curBlock->values)[i];
        (tmpBlock.header.size)++;
    }

    return 0;
}

MERCURY_NAMESPACE_END(core);
#endif //MERCURY_HASH_TABLE_H_
