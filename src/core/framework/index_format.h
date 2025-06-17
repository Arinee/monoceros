/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     index_format.h
 *   \author   Hechong.xyf
 *   \date     Nov 2017
 *   \version  1.0.0
 *   \brief    Interface of Mercury Index Format
 */

#ifndef __MERCURY_INDEX_FORMAT_H__
#define __MERCURY_INDEX_FORMAT_H__

#include <cstring>
#include <string>

namespace mercury {

/*! Index Format
 */
struct IndexFormat
{
    /*! Version number of format
     */
    enum
    {
        VERSION = 0x0001
    };

    /*! Index Format Segment Meta
     */
    struct SegmentMeta
    {
        uint8_t segment_id[64];
        uint8_t reserved_[8];
        uint64_t data_index;
        uint64_t data_size;
        uint32_t data_crc;
        uint32_t padding_size;

        //! Check the crc value of segment data
        bool check(const uint8_t *payload) const;

        //! Retrieve id of segment
        inline const char *getSegmentId(void) const;

        //! Retrieve data pointer
        inline const uint8_t *getData(const uint8_t *payload) const;

        //! Retrieve data pointer
        inline uint8_t *getData(uint8_t *payload) const;

        //! Retrieve data index
        inline uint64_t getDataIndex(void) const;

        //! Retrieve data size
        inline uint64_t getDataSize(void) const;

        //! Retrieve data crc
        inline uint32_t getDataCrc(void) const;

        //! Retrieve padding size
        inline uint32_t getPaddingSize(void) const;

        //! Set id of segment
        inline void setSegmentId(const char *name);

        //! Set data index of segment
        inline void setDataIndex(uint64_t index);

        //! Set data size of segment
        inline void setDataSize(uint64_t size);

        //! Set data crc of segment
        inline void setDataCrc(uint32_t crc);

        //! Set padding size of segment
        inline void setPaddingSize(uint32_t size);
    };

    static_assert(sizeof(SegmentMeta) % 32 == 0,
                  "SegmentMeta must be aligned with 32bits");

    /*! Index Format Meta
     */
    struct Meta
    {
        uint32_t head_crc;
        uint32_t payload_crc;
        uint32_t version;
        uint32_t margic;
        uint64_t revision;
        uint64_t tags;

        // total_size = meta_size + segments_size + payload_size
        // segment_count = segments_size / segment_meta_size
        uint16_t meta_size;
        uint16_t segment_meta_size;
        uint32_t segments_size;
        uint64_t payload_size;

        // Time stamp
        uint64_t setup_time;
        uint64_t update_time;

        //! Retrieve payload pointer
        inline const uint8_t *getPayload(void) const;

        //! Retrieve payload pointer
        inline uint8_t *getPayload(void);

        //! Retrieve payload size
        inline uint64_t getPayloadSize(void) const;

        //! Retrieve margic
        inline uint32_t getMargic(void) const;

        //! Retrieve segment count
        inline size_t getSegmentCount(void) const;

        //! Set size of payload
        inline void setPayloadSize(uint64_t size);

        //! Set margic of meta
        inline void setMargic(uint32_t val);

        //! Refresh meta information
        void refresh(void);

        //! Check the crc value of meta
        bool check(void) const;

        //! Update crc value of head
        void updateHeadCrc(void);

        //! Update crc value of payload
        void updatePayloadCrc(const void *payload, uint64_t len);

        //! Retrieve segment via index
        const SegmentMeta *getSegment(size_t index) const;

        //! Retrieve segment via index
        SegmentMeta *getSegment(size_t index);

        //! Retrieve segment via name
        const SegmentMeta *getSegment(const char *segid) const;

        //! Retrieve segment via name
        SegmentMeta *getSegment(const char *segid);
    };

    static_assert(sizeof(Meta) % 32 == 0, "Meta must be aligned with 32bits");

    //! Setup meta structure in buffer
    static Meta *SetupMeta(uint16_t segcount, std::string *buf);

    //! Update meta structure in buffer
    static Meta *UpdateMeta(void *data, size_t len);

    //! Update meta structure in buffer
    static Meta *UpdateMeta(std::string *buf)
    {
        return UpdateMeta(const_cast<char *>(buf->data()), buf->size());
    }

    //! Convert a data pointer to meta pointer
    static Meta *Cast(void *data, size_t len);

    //! Convert a data pointer to meta pointer
    static const Meta *Cast(const void *data, size_t len);

    //! Refresh meta information
    static bool Refresh(void *data, size_t len);

    //! Refresh meta information
    static bool Refresh(std::string *buf)
    {
        return IndexFormat::Refresh(const_cast<char *>(buf->data()),
                                    buf->size());
    }
};

//! Retrieve id of segment
const char *IndexFormat::SegmentMeta::getSegmentId(void) const
{
    return reinterpret_cast<const char *>(this->segment_id);
}

//! Retrieve data pointer
const uint8_t *IndexFormat::SegmentMeta::getData(const uint8_t *payload) const
{
    return (payload + this->data_index);
}

//! Retrieve data pointer
uint8_t *IndexFormat::SegmentMeta::getData(uint8_t *payload) const
{
    return (payload + this->data_index);
}

//! Retrieve data index
uint64_t IndexFormat::SegmentMeta::getDataIndex(void) const
{
    return (this->data_index);
}

//! Retrieve data size
uint64_t IndexFormat::SegmentMeta::getDataSize(void) const
{
    return (this->data_size);
}

uint32_t IndexFormat::SegmentMeta::getDataCrc(void) const
{
    return (this->data_crc);
}

//! Retrieve padding size
uint32_t IndexFormat::SegmentMeta::getPaddingSize(void) const
{
    return (this->padding_size);
}

//! Set id of segment
void IndexFormat::SegmentMeta::setSegmentId(const char *name)
{
    std::strncpy(reinterpret_cast<char *>(this->segment_id), name,
                 sizeof(this->segment_id) - 1);
}

//! Set data index of segment
void IndexFormat::SegmentMeta::setDataIndex(uint64_t index)
{
    this->data_index = index;
}

//! Set data size of segment
void IndexFormat::SegmentMeta::setDataSize(uint64_t size)
{
    this->data_size = size;
}

//! Set data crc of segment
void IndexFormat::SegmentMeta::setDataCrc(uint32_t crc)
{
    this->data_crc = crc;
}

//! Set padding size of segment
void IndexFormat::SegmentMeta::setPaddingSize(uint32_t size)
{
    this->padding_size = size;
}

//! Retrieve payload pointer
const uint8_t *IndexFormat::Meta::getPayload(void) const
{
    return (reinterpret_cast<const uint8_t *>(this) +
            (this->meta_size + this->segments_size));
}

//! Retrieve payload pointer
uint8_t *IndexFormat::Meta::getPayload(void)
{
    return (reinterpret_cast<uint8_t *>(this) +
            (this->meta_size + this->segments_size));
}

//! Retrieve payload size
uint64_t IndexFormat::Meta::getPayloadSize(void) const
{
    return (this->payload_size);
}

//! Retrieve margic
uint32_t IndexFormat::Meta::getMargic(void) const
{
    return (this->margic);
}

//! Retrieve segment count
size_t IndexFormat::Meta::getSegmentCount(void) const
{
    return static_cast<size_t>(this->segments_size / this->segment_meta_size);
}

//! Set size of payload
void IndexFormat::Meta::setPayloadSize(uint64_t size)
{
    this->payload_size = size;
}

//! Set margic of meta
void IndexFormat::Meta::setMargic(uint32_t val)
{
    this->margic = val;
}

} // namespace mercury

#endif // __MERCURY_INDEX_FORMAT_H__
