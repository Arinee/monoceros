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
#include "src/core/framework/index_logger.h"

MERCURY_NAMESPACE_BEGIN(core);

struct Header {
    size_t slotNum;
    size_t capacity;
    char padding[16];
};

struct SlotInfo {
    size_t codeCount;
    size_t startIndex; // Points to the first code in PackedCodeList
    size_t endIndex;   // Points to the last code in PackedCodeList
    char padding[8];
};

struct PackedCodeList {
    uint8_t* packedCodes; // Raw pointer to the packed codes
};

class FastScanIndex {
public:
    class PackedCodeIterator {
    public:
        PackedCodeIterator(const uint8_t* startPtr, size_t codeCount)
            : _startPtr(startPtr), _curIndex(0), _codeCount(codeCount) {}

        bool hasNext() const {
            return _curIndex < _codeCount;
        }

        const uint8_t& next() {
            if (!hasNext()) {
                throw std::out_of_range("No more codes available");
            }
            return _startPtr[_curIndex++];
        }

        const uint8_t* getStartPointer() const {
            return _startPtr;
        }

        size_t getCodeCount() const {
            return _codeCount;
        }

    private:
        const uint8_t* _startPtr;
        size_t _curIndex;
        size_t _codeCount;
    };

    void addCode(int32_t slotId, const uint8_t& code) {
        SlotInfo& slot = _indexSlots[slotId];
        _packedCodeList.packedCodes[slot.startIndex + slot.codeCount++] = code;
    }

    PackedCodeIterator search(int32_t slotId) const {
        const SlotInfo& slot = _indexSlots[slotId];
        if (slot.startIndex == -1UL) {
            return PackedCodeIterator(nullptr, 0); // Return an empty iterator if the slot is empty
        }
        const uint8_t* startPtr = &_packedCodeList.packedCodes[slot.startIndex];
        return PackedCodeIterator(startPtr, slot.codeCount);
    }

    void PrintStats() const {
        std::cout << "slot num: " << _pHeader->slotNum << std::endl;
        for (size_t i = 0; i < _pHeader->slotNum; ++i) {
            const SlotInfo& slot = _indexSlots[i];
            std::cout << "slot i: " << i << ", codeCount: " << slot.codeCount << std::endl;

            PackedCodeIterator it = search(i);
            std::cout << "PackedCodes: ";
            while (it.hasNext()) {
                uint8_t code = it.next();
                std::cout << unsigned(code) << " ";
            }
            std::cout << std::endl;
        }
    }

    bool create(void *pBase, size_t slotNum, const std::vector<size_t>& codeCounts)
    {
        assert(pBase != nullptr);
        assert(slotNum > 0);
        assert(codeCounts.size() == slotNum); // Ensure the code counts match the number of slots

        _pBase = static_cast<char*>(pBase);

        size_t capacity = calcSize(slotNum, codeCounts);

        // Optionally initialize the memory to zeros
        memset(_pBase, 0, capacity);

        _pHeader = reinterpret_cast<Header*>(_pBase);
        _pHeader->slotNum = slotNum;
        _pHeader->capacity = capacity;

        // Initialize the index slots with the correct start and end indices
        _indexSlots = reinterpret_cast<SlotInfo*>(_pBase + sizeof(Header));
        size_t currentIndex = 0;
        for (size_t i = 0; i < slotNum; ++i) {
            _indexSlots[i].codeCount = 0;
            _indexSlots[i].startIndex = currentIndex;
            _indexSlots[i].endIndex = currentIndex + codeCounts[i] - 1;
            currentIndex += codeCounts[i];
        }

        // Point the packed codes to the correct location
        _packedCodeList.packedCodes =
            reinterpret_cast<uint8_t*>(_pBase + sizeof(Header) + slotNum * sizeof(SlotInfo));
        
        return true;
    }

    bool load(void *pBase, size_t len)
    {
        assert(pBase != nullptr);
        _pBase = static_cast<char*>(pBase);
        _pHeader = reinterpret_cast<Header*>(pBase);

        if ((size_t)_pHeader->capacity != len) {
            std::cerr << "file size in header is not equal to real file size" << std::endl;
            return false;
        }

        // Set pointers to the correct locations in the memory block
        _indexSlots = reinterpret_cast<SlotInfo*>(_pBase + sizeof(Header));
        _packedCodeList.packedCodes =
            reinterpret_cast<uint8_t*>(_pBase + sizeof(Header) + _pHeader->slotNum * sizeof(SlotInfo));
        if (isAligned32(_packedCodeList.packedCodes)) {
            LOG_INFO("fastscan index load success!");
        } else {
            LOG_ERROR("failed to load fastscan index!");
            return false;
        }

        // PrintStats();
        
        return true;
    }

    static size_t calcSize(size_t slotNum, const std::vector<size_t>& codeCounts) {
        // Calculate size of the Header
        size_t totalSize = sizeof(Header);

        // Calculate size of the SlotInfo array
        totalSize += slotNum * sizeof(SlotInfo);

        // Calculate size of the PackedCodeList
        for (size_t codeCount : codeCounts) {
            totalSize += codeCount * sizeof(uint8_t);
        }

        return totalSize;
    }

    static size_t ROUNDUP32(size_t a) {
        return (a + 32UL - 1) / 32UL * 32UL;
    }

    /* extract the column starting at (i, j)
    * from packed matrix src of size (m, n)*/
    template <typename T, class TA>
    static void get_matrix_column(
            T* src,
            size_t m,
            size_t n,
            int64_t i,
            int64_t j,
            TA& dest) {
        for (uint64_t k = 0; k < dest.size(); k++) {
            if (k + i >= 0 && k + i < m) {
                dest[k] = src[(k + i) * n + j];
            } else {
                dest[k] = 0;
            }
        }
    }

    const void* GetBasePtr() const { return static_cast<const void*>(_pBase); }

    const Header *getHeader() const {
        return _pHeader;
    }

    Header *getHeader() {
        return _pHeader;
    }

    const SlotInfo *getSlotinfo() const {
        return _indexSlots;
    }

    SlotInfo *getSlotinfo() {
        return _indexSlots;
    }

private:
    char *_pBase;
    Header *_pHeader;
    SlotInfo* _indexSlots; 
    PackedCodeList _packedCodeList;
};

MERCURY_NAMESPACE_END(core);
