#ifndef __MERCURY_CORE_VISIT_LIST_H__
#define __MERCURY_CORE_VISIT_LIST_H__

#include <limits>
#include <cstdint>
#include "src/core/utils/deletemap.h"

MERCURY_NAMESPACE_BEGIN(core);

template <typename T = uint8_t>
class VisitList
{
public:
    VisitList(void)
        : _maxDocCnt(0), max_scan_num_(0), visited_num_(0), cur_num_(1)
    {
    }

    ~VisitList(void) {}

    int init(uint64_t maxDocCnt, uint64_t maxScanNum)
    {
        _maxDocCnt = maxDocCnt;
        max_scan_num_ = maxScanNum;
        visited_num_ = 0;

        arr_.resize(maxDocCnt);
        cur_num_ = 1;

        return 0;
    }

    // attention:call visited to see whether need set
    void setVisited(docid_t idx)
    {
        arr_.at(idx) = true;
        visited_num_++;
    }

    uint64_t getVisitedNum()
    {
        return visited_num_;
    }

    uint64_t getMaxDocCnt()
    {
        return _maxDocCnt;
    }

    bool visited(docid_t idx)
    {
        return arr_.at(idx);
    }

    bool needBreak(void)
    {
        return visited_num_ >= max_scan_num_;
    }

    void reset(uint64_t maxDocId)
    {
        arr_.assign(arr_.size(), false);
        return;
    }

private:
    uint64_t _maxDocCnt;
    uint64_t max_scan_num_;
    uint64_t visited_num_;
    T cur_num_;
    std::vector<bool> arr_;
};

class VisitMap
{
public:
    VisitMap(void)
    : delete_map_(nullptr),
      arr_(nullptr),
      max_scan_num_(0),
      visited_num_(0),
      start_idx_(std::numeric_limits<docid_t>::max()),
      end_idx_(0),
      visited_arr_(nullptr)
    {
    }

    ~VisitMap(void)
    {
        delete delete_map_;
        delete_map_ = nullptr;

        delete [] arr_;
        arr_ = nullptr;

        delete [] visited_arr_;
        visited_arr_ = nullptr;
    }

    int init(uint64_t maxDocCnt, uint64_t maxScanNum)
    {
        max_scan_num_ = maxScanNum;

        if (delete_map_ == nullptr) {
            delete_map_ = new (std::nothrow) DeleteMap(maxDocCnt);
            if (delete_map_ == nullptr) {
                LOG_ERROR("New DeleteMap failed");
                return IndexError_NoMemory;
            }
        }

        if (arr_ != nullptr) {
            delete [] arr_;
        }

        delete_map_->unmount();

        uint64_t byteSize = delete_map_->size();
        arr_ = new (std::nothrow) char[byteSize];
        if (arr_ == nullptr) {
            LOG_ERROR("New byte size[%lu] for arr_ failed", byteSize);
            return IndexError_NoMemory;
        }
        
        int iRet = delete_map_->mount(arr_, byteSize, true);
        if (iRet != 0) {
            LOG_ERROR("Mount delete map failed");
            return iRet;
        }

        if (visited_arr_ != nullptr) {
            delete [] visited_arr_;
        }
        
        visited_arr_ = new (std::nothrow) docid_t[max_scan_num_];
        if (visited_arr_ == nullptr) {
            LOG_ERROR("New memory for visited array failed");
            return IndexError_NoMemory;
        }
        visited_num_ = 0U;

        return 0;
    }
    //attention:call visited to see whether need set
    void setVisited(docid_t idx)
    {
        delete_map_->setInvalid(idx);
        if (idx < start_idx_) {
            start_idx_ = idx;
        }

        if (idx > end_idx_) {
            end_idx_ = idx;
        }
        
        if (visited_num_ < max_scan_num_) {
            visited_arr_[visited_num_++] = idx;
        }

        return;
    }

    bool visited(docid_t idx)
    {
        return delete_map_->isInvalid(idx);
    }

    bool needBreak(void)
    {
        return visited_num_ >= max_scan_num_;
    }

    void reset(void)
    {
        //bit map has never been set
        if (end_idx_ == 0) {
            return;
        }

        docid_t span = (end_idx_ - start_idx_) >> DeleteMap::SLOT_SIZE_BIT_NUM;
        //if visited_num_ is more than 4 times of span
        if (span < (visited_num_ << 2)) {
            delete_map_->setAllValid(start_idx_, end_idx_);
        } else {
            for (uint32_t i = 0; i < visited_num_; ++i) {
                delete_map_->setAllValid(visited_arr_[i]);
            }
        }

        start_idx_ = std::numeric_limits<docid_t>::max();
        end_idx_ = 0;
        visited_num_ = 0;

        return;
    }

private:
    DeleteMap *delete_map_;
    char *arr_;
    uint64_t max_scan_num_;
    uint64_t visited_num_;
    docid_t start_idx_;
    docid_t end_idx_;
    docid_t *visited_arr_;
};

MERCURY_NAMESPACE_END(core);

#endif //__MERCURY_CORE_VISIT_LIST_H__
