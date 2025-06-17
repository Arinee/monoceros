/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     base_index_provider.h
 *   \author   yunheng@xiaohongshu.com
 *   \date     Jan 2019
 *   \version  1.0.0
 *   \brief    INDEX PROVIDER
 */

#ifndef __MERCURY_INDEX_BASE_INDEX_PROVIDER_H__
#define __MERCURY_INDEX_BASE_INDEX_PROVIDER_H__

#include <memory>
#include "framework/utility/mmap_file.h"
#include "framework/index_framework.h"
#include "common/common_define.h"
#include "index.h"

namespace mercury {

// Must load, before online service using
class BaseIndexProvider 
{
public:
    BaseIndexProvider()
        : _incrId(0),
        _incrSegmentDocNum(0)
    {}
    virtual ~BaseIndexProvider()
    {}

    virtual bool init(size_t incrSegmentDocNum, 
            const std::string &incrSegmentPath,
            std::shared_ptr<IndexParams> index_params_pointer);
    virtual bool load(IndexStorage::Handler::Pointer &&handlerPtr, std::shared_ptr<Index> segmentPtr); 
    virtual bool unload();

    const IndexMeta * get_index_meta() const {
        if (likely(_segments.size() > 0)) {
            // get index meta from first segment
            return _segments[0]->get_index_meta();
        }
        return nullptr;
    }

    gloid_t addVector(key_t key, const void *feature, size_t len);
    bool deleteVector(key_t key);

    // profile assistant
    key_t getPK(gloid_t gloid) const;
    const void *getFeature(gloid_t gloid) const;

    const std::vector<Index::Pointer>& get_segment_list() const
    {
        return _segments;
    }

private:
    BaseIndexProvider(const BaseIndexProvider &BaseIndexProvider) = delete;
    BaseIndexProvider &operator=(const BaseIndexProvider &BaseIndexProvider) = delete;

    virtual bool createSegment(std::shared_ptr<Index> segmentPtr);

protected:
    std::mutex _addLock;
    std::vector<std::shared_ptr<Index>> _segments;
    std::shared_ptr<Index> _lastSegment;
    std::vector<std::unique_ptr<mercury::MMapFile>> _incrFileHolder;
    std::string _incrSegmentPath;
    std::shared_ptr<IndexParams> index_params_pointer_;

    docid_t _incrId;
    size_t _incrSegmentDocNum;
};

class FaissIndexProvider : public BaseIndexProvider {};

} // namespace mercury

#endif // __MERCURY_INDEX_BASE_INDEX_PROVIDER_H__
