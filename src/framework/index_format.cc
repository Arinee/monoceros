/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     index_format.cc
 *   \author   Hechong.xyf
 *   \date     Nov 2017
 *   \version  1.0.0
 *   \brief    Implementation of Mercury Index Format
 */

#include "index_format.h"
#include "utility/crypto.h"
#include "utility/time_helper.h"

namespace mercury {

bool IndexFormat::SegmentMeta::check(const uint8_t *payload) const
{
    return (this->data_crc == Crypto::Crc32c(0, payload + this->data_index,
                                             (size_t)this->data_size));
}

void IndexFormat::Meta::refresh(void)
{
    uint32_t head_size = this->meta_size + this->segments_size;
    this->payload_crc =
        Crypto::Crc32c(0, reinterpret_cast<uint8_t *>(this) + head_size,
                       (size_t)(this->payload_size));
    this->update_time = Realtime::Seconds();
    this->head_crc = 0;
    this->head_crc = Crypto::Crc32c(0, this, head_size);
}

bool IndexFormat::Meta::check(void) const
{
    const uint8_t *head = reinterpret_cast<const uint8_t *>(this);
    uint32_t head_size = this->meta_size + this->segments_size;
    return (this->head_crc == Crypto::Crc32c(this->head_crc, head, head_size) &&
            this->payload_crc == Crypto::Crc32c(0, head + head_size,
                                                (size_t)this->payload_size));
}

void IndexFormat::Meta::updateHeadCrc(void)
{
    uint32_t head_size = this->meta_size + this->segments_size;
    this->update_time = Realtime::Seconds();
    this->head_crc = 0;
    this->head_crc = Crypto::Crc32c(0, this, head_size);
}

void IndexFormat::Meta::updatePayloadCrc(const void *payload, uint64_t len)
{
    this->payload_crc = Crypto::Crc32c(this->payload_crc, payload, len);
    this->update_time = Realtime::Seconds();
}

const IndexFormat::SegmentMeta *
IndexFormat::Meta::getSegment(size_t index) const
{
    const SegmentMeta *segment_meta = NULL;

    if (index < this->getSegmentCount()) {
        segment_meta = reinterpret_cast<const SegmentMeta *>(
            reinterpret_cast<const uint8_t *>(this) +
            (this->meta_size + this->segment_meta_size * index));
    }
    return segment_meta;
}

IndexFormat::SegmentMeta *IndexFormat::Meta::getSegment(size_t index)
{
    SegmentMeta *segment_meta = NULL;

    if (index < this->getSegmentCount()) {
        segment_meta = reinterpret_cast<SegmentMeta *>(
            reinterpret_cast<uint8_t *>(this) +
            (this->meta_size + this->segment_meta_size * index));
    }
    return segment_meta;
}

IndexFormat::SegmentMeta *IndexFormat::Meta::getSegment(const char *segid)
{
    uint8_t *head = reinterpret_cast<uint8_t *>(this);
    uint64_t segment_index = this->meta_size;
    uint64_t segment_end = segment_index + this->segments_size;

    for (; segment_index != segment_end;
         segment_index += this->segment_meta_size) {
        SegmentMeta *segment_meta =
            reinterpret_cast<SegmentMeta *>(head + segment_index);

        if (!std::strncmp(segid, segment_meta->getSegmentId(),
                          sizeof(segment_meta->segment_id))) {
            return segment_meta;
        }
    }
    return NULL;
}

const IndexFormat::SegmentMeta *
IndexFormat::Meta::getSegment(const char *segid) const
{
    const uint8_t *head = reinterpret_cast<const uint8_t *>(this);
    uint64_t segment_index = this->meta_size;
    uint64_t segment_end = segment_index + this->segments_size;

    for (; segment_index != segment_end;
         segment_index += this->segment_meta_size) {
        const SegmentMeta *segment_meta =
            reinterpret_cast<const SegmentMeta *>(head + segment_index);
        if (!std::strncmp(segid, segment_meta->getSegmentId(),
                          sizeof(segment_meta->segment_id))) {
            return segment_meta;
        }
    }
    return NULL;
}

IndexFormat::Meta *IndexFormat::SetupMeta(uint16_t segcount, std::string *buf)
{
    uint32_t head_size;

    segcount = ((segcount + 3) >> 2) << 2;
    head_size = sizeof(Meta) + sizeof(SegmentMeta) * segcount;
    buf->clear();
    buf->resize(head_size);

    // Update meta information
    Meta *meta = reinterpret_cast<Meta *>(const_cast<char *>(buf->data()));
    meta->head_crc = 0;
    meta->payload_crc = 0;
    meta->version = IndexFormat::VERSION;
    meta->margic = static_cast<uint32_t>(rand());
    meta->revision = 0;
    meta->tags = 0;
    meta->meta_size = sizeof(Meta);
    meta->segment_meta_size = sizeof(SegmentMeta);
    meta->segments_size = sizeof(SegmentMeta) * segcount;
    meta->payload_size = 0;
    meta->update_time = meta->setup_time = Realtime::Seconds();
    meta->head_crc = Crypto::Crc32c(0, meta, head_size);
    return meta;
}

IndexFormat::Meta *IndexFormat::UpdateMeta(void *data, size_t len)
{
    Meta *meta = IndexFormat::Cast(data, len);
    platform_null_if_false(meta);

    uint32_t head_size = meta->meta_size + meta->segments_size;
    size_t orig_size = head_size + (size_t)meta->payload_size;
    size_t extend_size = len - orig_size;
    if (extend_size) {
        meta->payload_crc = Crypto::Crc32c(
            meta->payload_crc, reinterpret_cast<uint8_t *>(meta) + orig_size,
            extend_size);
        meta->payload_size += extend_size;
    }
    meta->update_time = Realtime::Seconds();
    meta->head_crc = 0;
    meta->head_crc = Crypto::Crc32c(0, meta, head_size);
    return meta;
}

const IndexFormat::Meta *IndexFormat::Cast(const void *data, size_t len)
{
    const Meta *meta = reinterpret_cast<const Meta *>(data);
    platform_null_if_false(
        (sizeof(Meta) < len) && (sizeof(Meta) <= meta->meta_size) &&
        (sizeof(SegmentMeta) <= meta->segment_meta_size) &&
        ((meta->segments_size % meta->segment_meta_size) == 0) &&
        ((meta->meta_size + meta->segments_size + meta->payload_size) <= len));
    return meta;
}

IndexFormat::Meta *IndexFormat::Cast(void *data, size_t len)
{
    Meta *meta = reinterpret_cast<Meta *>(data);
    platform_null_if_false(
        (sizeof(Meta) < len) && (sizeof(Meta) <= meta->meta_size) &&
        (sizeof(SegmentMeta) <= meta->segment_meta_size) &&
        ((meta->segments_size % meta->segment_meta_size) == 0) &&
        ((meta->meta_size + meta->segments_size + meta->payload_size) <= len));
    return meta;
}

bool IndexFormat::Refresh(void *data, size_t len)
{
    Meta *meta = IndexFormat::Cast(data, len);
    platform_false_if_false(meta);
    meta->refresh();
    return true;
}

} // namespace mercury
