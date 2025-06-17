/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     index_package.cc
 *   \author   Hechong.xyf
 *   \date     Mar 2018
 *   \version  1.0.0
 *   \brief    Implementation of Mercury Index Package
 */

#include "index_package.h"
#include "index_format.h"

namespace mercury {

static inline size_t CalcPaddingSize(size_t size)
{
    const size_t bound = 32;
    return ((size + bound - 1) / bound * bound - size);
}

bool IndexPackage::load(const IndexStorage::Handler::Pointer &handle,
                        bool checksum)
{
    const void *data = nullptr;
    size_t len = handle->read(&data, handle->size());
    if (len == 0) {
        return false;
    }

    const IndexFormat::Meta *meta = IndexFormat::Cast(data, len);
    if ((!meta) || (checksum && !meta->check())) {
        return false;
    }

    for (size_t i = 0, seg_count = meta->getSegmentCount(); i < seg_count;
         ++i) {
        const IndexFormat::SegmentMeta *segment = meta->getSegment(i);
        size_t data_size = static_cast<size_t>(segment->getDataSize());

        if (data_size &&
            segment->getDataIndex() + data_size <= meta->getPayloadSize()) {
            _vec.emplace_back(segment->getSegmentId(),
                              segment->getData(meta->getPayload()),
                              (size_t)data_size);
        }
    }
    return true;
}

bool IndexPackage::dump(const std::string &path,
                        const IndexStorage::Pointer &stg, bool checksum)
{
    size_t total_size = 0;
    for (auto iter = _vec.begin(); iter != _vec.end(); ++iter) {
        total_size +=
            (iter->getDataSize() + CalcPaddingSize(iter->getDataSize()));
    }

    std::string buffer;
    IndexFormat::Meta *meta = IndexFormat::SetupMeta(_vec.size(), &buffer);
    total_size += buffer.size();

    IndexStorage::Handler::Pointer handle = stg->create(path, total_size);
    if (!handle) {
        return false;
    }

    std::string padding;
    uint64_t data_index = 0;
    for (size_t i = 0; i < _vec.size(); ++i) {
        IndexFormat::SegmentMeta *dst = meta->getSegment(i);
        const IndexPackage::Segment &src = _vec[i];
        size_t padding_size = CalcPaddingSize(src.getDataSize());

        dst->setSegmentId(src.getSegmentId().c_str());
        dst->setDataIndex(data_index);
        dst->setDataSize(src.getDataSize());
        dst->setDataCrc(src.getDataCrc());
        dst->setPaddingSize(padding_size);

        if (checksum) {
            meta->updatePayloadCrc(src.getData(), src.getDataSize());

            if (padding_size) {
                padding.resize(padding_size);
                meta->updatePayloadCrc(padding.data(), padding.size());
            }
        }

        // Next data index
        data_index += src.getDataSize() + padding_size;
    }
    meta->setPayloadSize(data_index);
    meta->updateHeadCrc();

    // Write head into storage
    if (handle->write(buffer.data(), buffer.size()) != buffer.size()) {
        return false;
    }

    // Write segments
    for (size_t i = 0; i < _vec.size(); ++i) {
        const IndexPackage::Segment &src = _vec[i];
        size_t padding_size = CalcPaddingSize(src.getDataSize());

        if (handle->write(src.getData(), src.getDataSize()) !=
            src.getDataSize()) {
            return false;
        }

        // Write the padding if need
        if (padding_size) {
            padding.resize(padding_size);
            if (handle->write(padding.data(), padding.size()) !=
                padding.size()) {
                return false;
            }
        }
    }
    return true;
}

void IndexPackage::clear(void)
{
    _vec.clear();
}

size_t IndexPackage::count(void) const
{
    return _vec.size();
}

const IndexPackage::Segment *IndexPackage::get(size_t index) const
{
    return (index < _vec.size() ? &_vec[index] : nullptr);
}

IndexPackage::Segment *IndexPackage::get(size_t index)
{
    return (index < _vec.size() ? &_vec[index] : nullptr);
}

const IndexPackage::Segment *IndexPackage::get(const std::string &segid) const
{
    for (auto iter = _vec.begin(); iter != _vec.end(); ++iter) {
        if (iter->getSegmentId() == segid) {
            return &(*iter);
        }
    }
    return nullptr;
}

IndexPackage::Segment *IndexPackage::get(const std::string &segid)
{
    for (auto iter = _vec.begin(); iter != _vec.end(); ++iter) {
        if (iter->getSegmentId() == segid) {
            return &(*iter);
        }
    }
    return nullptr;
}

void IndexPackage::push(const IndexPackage::Segment &seg)
{
    _vec.push_back(seg);
}

void IndexPackage::emplace(const std::string &segid, const void *data,
                           size_t data_size)
{
    _vec.emplace_back(segid, data, data_size);
}

void IndexPackage::emplace(std::string &&segid, const void *data,
                           size_t data_size)
{
    _vec.emplace_back(std::forward<std::string>(segid), data, data_size);
}

bool IndexPackage::Touch(const std::string &path,
                         const IndexStorage::Pointer &stg,
                         const std::map<std::string, size_t> &stab)
{
    size_t total_size = 0;
    for (auto iter = stab.begin(); iter != stab.end(); ++iter) {
        total_size += iter->second + CalcPaddingSize(iter->second);
    }

    std::string buffer;
    IndexFormat::Meta *meta = IndexFormat::SetupMeta(stab.size(), &buffer);
    total_size += buffer.size();

    IndexStorage::Handler::Pointer handle = stg->create(path, total_size);
    if (!handle) {
        return false;
    }

    size_t segment_index = 0;
    uint64_t data_index = 0;
    for (const auto &it : stab) {
        IndexFormat::SegmentMeta *dst = meta->getSegment(segment_index);
        size_t padding_size = CalcPaddingSize(it.second);

        dst->setSegmentId(it.first.c_str());
        dst->setDataIndex(data_index);
        dst->setDataSize(it.second);
        dst->setDataCrc(0);
        dst->setPaddingSize(padding_size);

        // Next data index
        data_index += it.second + padding_size;
        ++segment_index;
    }
    meta->setPayloadSize(data_index);
    meta->updateHeadCrc();

    // Write head into storage
    if (handle->write(buffer.data(), buffer.size()) != buffer.size()) {
        return false;
    }

    std::string padding;
    padding.resize(8192u);

    size_t segments_size = total_size - buffer.size();
    size_t padding_count = segments_size / padding.size();

    // Write segments
    for (size_t i = 0; i < padding_count; ++i) {
        if (handle->write(padding.data(), padding.size()) != padding.size()) {
            return false;
        }
    }
    padding.resize(segments_size % padding.size());
    if (padding.size()) {
        if (handle->write(padding.data(), padding.size()) != padding.size()) {
            return false;
        }
    }
    return true;
}

} // namespace mercury
