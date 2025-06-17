/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     index_package.h
 *   \author   Hechong.xyf
 *   \date     Feb 2018
 *   \version  1.0.0
 *   \brief    Interface of Mercury Index Package
 */

#ifndef __MERCURY_INDEX_PACKAGE_H__
#define __MERCURY_INDEX_PACKAGE_H__

#include "index_storage.h"
#include <map>
#include <string>
#include <vector>

namespace mercury {

/*! Index Package
 */
class IndexPackage
{
public:
    /*! Index Package Segment
     */
    class Segment
    {
    public:
        //! Constructor
        Segment(void)
            : _segment_id(), _data(nullptr), _data_size(0), _data_crc(0)
        {
        }

        //! Constructor
        Segment(const Segment &rhs)
            : _segment_id(rhs._segment_id), _data(rhs._data),
              _data_size(rhs._data_size), _data_crc(rhs._data_crc)
        {
        }

        //! Constructor
        Segment(const std::string &segid, const void *data, size_t data_size)
            : _segment_id(segid), _data(const_cast<void *>(data)),
              _data_size(data_size), _data_crc(0)
        {
        }

        //! Constructor
        Segment(std::string &&segid, const void *data, size_t data_size)
            : _segment_id(std::forward<std::string>(segid)),
              _data(const_cast<void *>(data)), _data_size(data_size),
              _data_crc(0)
        {
        }

        //! Assignment
        Segment &operator=(const Segment &rhs)
        {
            _segment_id = rhs._segment_id;
            _data = rhs._data;
            _data_size = rhs._data_size;
            _data_crc = rhs._data_crc;
            return *this;
        }

        //! Retrieve id of segment
        std::string &getSegmentId(void)
        {
            return _segment_id;
        }

        //! Retrieve id of segment
        const std::string &getSegmentId(void) const
        {
            return _segment_id;
        }

        //! Retrieve data of segment
        void *getData(void)
        {
            return _data;
        }

        //! Retrieve data of segment
        const void *getData(void) const
        {
            return _data;
        }

        //! Retrieve data size of segment
        size_t getDataSize(void) const
        {
            return _data_size;
        }

        //! Retrieve data crc of segment
        uint32_t getDataCrc(void) const
        {
            return _data_crc;
        }

        //! Set id of segment
        void setSegmentId(const std::string &segid)
        {
            _segment_id = segid;
        }

        //! Set data of segment
        void setData(const void *data, size_t size)
        {
            _data = const_cast<void *>(data);
            _data_size = size;
        }

        //! Set data crc of segment
        void setDataCrc(uint32_t crc)
        {
            _data_crc = crc;
        }

    private:
        std::string _segment_id;
        void *_data;
        size_t _data_size;
        uint32_t _data_crc;
    };

    //! Constructor
    IndexPackage(void) : _vec() {}

    //! Constructor
    IndexPackage(const IndexPackage &rhs) : _vec(rhs._vec) {}

    //! Constructor
    IndexPackage(IndexPackage &&rhs) : _vec(std::move(rhs._vec)) {}

    //! Assignment
    IndexPackage &operator=(const IndexPackage &rhs)
    {
        _vec = rhs._vec;
        return *this;
    }

    //! Assignment
    IndexPackage &operator=(IndexPackage &&rhs)
    {
        _vec = std::move(rhs._vec);
        return *this;
    }

    //! Load all segments from storage handler
    bool load(const IndexStorage::Handler::Pointer &handle,
              bool checksum = true);

    //! Dump all segments into storage
    bool dump(const std::string &path, const IndexStorage::Pointer &stg,
              bool checksum = true);

    //! Clear the package
    void clear(void);

    //! Retrieve count of segments
    size_t count(void) const;

    //! Retrieve segment via index
    const IndexPackage::Segment *get(size_t index) const;

    //! Retrieve segment via index
    IndexPackage::Segment *get(size_t index);

    //! Retrieve segment via name
    const IndexPackage::Segment *get(const std::string &segid) const;

    //! Retrieve segment via name
    IndexPackage::Segment *get(const std::string &segid);

    //! Push a segment into package
    void push(const IndexPackage::Segment &seg);

    //! Emplace a segment into package
    void emplace(const std::string &segid, const void *data, size_t data_size);

    //! Emplace a segment into package
    void emplace(std::string &&segid, const void *data, size_t data_size);

    //! Touch an empty package into storage
    static bool Touch(const std::string &path, const IndexStorage::Pointer &stg,
                      const std::map<std::string, size_t> &stab);

private:
    std::vector<IndexPackage::Segment> _vec;
};

} // namespace mercury

#endif // __MERCURY_INDEX_PACKAGE_H__
