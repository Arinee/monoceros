#ifndef __FILE_INDEX_HOLDER_H__
#define __FILE_INDEX_HOLDER_H__

#include "framework/index_framework.h"
#include "framework/utility/thread_pool.h"
#include "utils/string_util.h"
#include <string>
#include <stdio.h>
#include <string.h>
#include <climits>
#include <assert.h>
#include <gflags/gflags.h>

#define _INVALID_KEY ULONG_MAX

DEFINE_uint32(txt_process_threads, 16, "txt process threads, 0 refer to default cpu cores");
DEFINE_uint32(txt_process_block_in_athread, 200ul, "txt process block in a thread");


namespace mercury {

/*!
 * File Index Holder
 *  framework will use VectorHolder in this way:
 *  for (iter = createIterator(); iter && iter->isValid(); iter->next()) {
 *      key = iter->key();
 *      data = iter->data();
 *  }
 */

class FloatVectorReader
{
public:
    typedef std::shared_ptr<FloatVectorReader> Pointer;

    FloatVectorReader(size_t d,
                    const std::string& firstSep,
                    const std::string& secondSep,
                    const bool catEnabled_ = false)
            : _fp(nullptr),
              _dimension(d),
              _firstSep(firstSep),
              _secondSep(secondSep),
              _catEnabled(catEnabled_)
    {
        _buf = new char[BUF_SIZE];
    }

    bool load(const std::string& filePath)
    {
        _fp = fopen(filePath.c_str(), "r");
        if (!_fp) {
            fprintf(stdout, "Open file(%s) error: %s\n",
                    filePath.c_str(), strerror(errno));
            return false;
        }

        if (!loadAllData()) {
            fprintf(stdout,"load all data error.\n");
            return false;
        }
        return true;
    }

    // default file format: key;value1,value2,...valueN
    bool loadAllData(void)
    {
        if (!_fp) {
            fprintf(stdout, "File handler is nullptr\n");
            return false;
        }
        size_t totalSize = getFileRowsNum();
        if (totalSize == 0) {
            fprintf(stdout, "file is empty.\n");
            return false;
        }
        _keys.resize(totalSize);
        _values.resize(totalSize);
        if (_catEnabled) {
            _cats.resize(totalSize);
        }

        uint32_t concurrency =
                FLAGS_txt_process_threads == 0 ?
                std::thread::hardware_concurrency() : FLAGS_txt_process_threads;
        ThreadPool threadPool(false, concurrency);
        fprintf(stdout, "expected total size: %lu, thread pool count: %lu, row num in a thread: %u\n",
                totalSize, threadPool.count(), FLAGS_txt_process_block_in_athread);

        size_t index = 0;
        std::vector<std::string> records;
        while (fgets(_buf, BUF_SIZE, _fp)) {
            std::string input(_buf);
            records.push_back(input);
            if (records.size() >= FLAGS_txt_process_block_in_athread) {
                size_t rowNum = records.size();
                std::vector<std::string> v;
                v.swap(records);
                Closure::Pointer task = Closure::New(
                        this,
                        &FloatVectorReader::processMultiRows,
                        std::move(v),
                        index);
                index += rowNum;
                threadPool.enqueue(task, true);
            }
        }
        if (!records.empty()) {
            size_t rowNum = records.size();
            std::vector<std::string> v;
            v.swap(records);
            Closure::Pointer task = Closure::New(
                    this,
                    &FloatVectorReader::processMultiRows,
                    std::move(v),
                    index);
            index += rowNum;
            threadPool.enqueue(task, true);
        }
        threadPool.wakeAll();
        threadPool.waitFinish();

        //remove invalid ones
        std::sort(_invalidIndex.begin(), _invalidIndex.end(), std::greater<decltype(_invalidIndex)::value_type>());
        for (const auto& e : _invalidIndex) {
            if (_catEnabled) _cats.erase(_cats.begin() + e);
            _keys.erase(_keys.begin() + e);
            _values.erase(_values.begin() + e);
        }
        fprintf(stdout, "expected totalNum: %lu, read Num: %lu, valid num: %zu \n", totalSize, index, _cats.size());
        return true;
    }

    void processMultiRows(std::vector<std::string> &&records, size_t index)
    {
        char firstSep = _firstSep.at(0);
        char secondSep = _secondSep.at(0);

        for (size_t i = 0; i < records.size(); ++i) {
            std::vector<float> data;
            const char* record = records[i].c_str();
            char* cursor = const_cast<char*>(record);
            char* endptr = nullptr;
            if (_catEnabled) {
                cat_t cat = strtoul(cursor, &endptr, 10);
                _cats[index + i] = cat;
                cursor = strchr(endptr, firstSep);
                if (!cursor) {
                    fprintf(stdout, "Record format error - cat: %s\n", record);
                    std::lock_guard<std::mutex> lk(_mtx);
                    _invalidIndex.push_back(index+i);
                    continue;
                }
                ++cursor;
            }
            key_t key = strtoul(cursor, &endptr, 10);
            cursor = strchr(endptr, firstSep);
            if (!cursor) {
                fprintf(stdout, "Record format error - key: %s\n", record);
                std::lock_guard<std::mutex> lk(_mtx);
                _invalidIndex.push_back(index+i);
                continue;
            }
            do {
                float f = strtof(cursor+1, &endptr);
                data.push_back(f);
            } while((cursor = strchr(endptr, secondSep)) != nullptr);
            if (data.size() < _dimension) {
                fprintf(stdout, "Record format error - data: %s\n", record);
                std::lock_guard<std::mutex> lk(_mtx);
                _invalidIndex.push_back(index+i);
                continue;
            }
            _keys[index + i] = key;
            _values[index + i] = std::move(data);
        }
    }

    size_t getFileRowsNum()
    {
        size_t num = 0;
        while (fgets(_buf, BUF_SIZE, _fp)) {
            ++num;
        }
        int ret = fseek(_fp, 0L, SEEK_SET);
        if (ret != 0) {
            fprintf(stdout, "fseek error: %s\n", strerror(errno));
        }
        return num;
    }

    virtual ~FloatVectorReader() {
        if (_fp) {
            fclose(_fp);
            _fp = nullptr;
        }
        if (_buf) {
            delete[] _buf;
        }
    }

    VectorHolder::Iterator::Pointer createIterator(void) const {
        return VectorHolder::Iterator::Pointer(new FloatVectorReader::Iterator(this));
    }

    class Iterator : public mercury::VectorHolder::Iterator
    {
    public:
        //! Constructor
        explicit Iterator(const FloatVectorReader* holder)
                : _catEnabled(holder->_catEnabled),
                _cats(holder->_cats),
                _keys(holder->_keys),
                _values(holder->_values),
                cursor(0)
        {
        }

        ~Iterator() override = default;

        //! Test if the iterator is valid
        bool isValid(void) const override
        {
            return cursor < _keys.size();
        }

        //! Retrieve primary key
        key_t key(void) const override
        {
            return _keys[cursor];
        }

        cat_t cat() const override {
            if (_catEnabled) {
                return _cats[cursor];
            } else {
                return INVALID_CAT_ID;
            }
        }

        //! Retrieve pointer of data
        const void* data() const override
        {
            return _values[cursor].data();
        }

        //! Next iterator
        void next(void) override
        {
            cursor++;
        }

        //! Reset the iterator
        void reset(void) override
        {
            cursor = 0;
        }
    private:
        const bool _catEnabled;
        const std::vector<cat_t>& _cats;
        const std::vector<key_t>& _keys;
        const std::vector<std::vector<float>>& _values;
        size_t cursor;
    };

protected:
    FILE *_fp;
    size_t _dimension;
    std::string _firstSep;
    std::string _secondSep;
    const bool _catEnabled = false;
    std::vector<cat_t> _cats;
    std::vector<key_t> _keys;
    std::vector<std::vector<float>> _values;
    char *_buf;
    std::vector<size_t> _invalidIndex;
    std::mutex _mtx;

    const static size_t BUF_SIZE = 40960;
};

class TxtFileHolder : public mercury::VectorHolder
{
public:
    typedef std::shared_ptr<TxtFileHolder> Pointer;

    TxtFileHolder(VectorHolder::FeatureType t,
                     size_t d,
                     const std::string &firstSep,
                     const std::string &secondSep,
                     const bool catEnabled_ = false)
            : _holderImpl(nullptr),
              _dimension(d),
              _type(t),
              _catEnabled(catEnabled_)
    {
        if (_type == IndexMeta::kTypeFloat) {
            _holderImpl = new FloatVectorReader(_dimension, firstSep, secondSep, _catEnabled);
        } else {
            fprintf(stderr, "Input type is not supported\n");
        }
    }

    bool isCatEnabled() const { return _catEnabled; }

    bool load(const std::string& filePath)
    {
        if (!_holderImpl) {
            return false;
        }
        return _holderImpl->load(filePath);
    }

    ~TxtFileHolder() override {
        delete _holderImpl;
    }

    //! Retrieve count of elements in holder
    size_t count() const override
    {
        return (size_t)-1;
    }

    //! Retrieve dimension
     size_t dimension() const override
    {
        return _dimension;
    }

    //! Retrieve type information
     VectorHolder::FeatureType type() const override
    {
        return _type;
    }

    //! Create a new iterator
     VectorHolder::Iterator::Pointer createIterator() const override
    {
        if (_holderImpl) {
            return _holderImpl->createIterator();
        } else {
            return VectorHolder::Iterator::Pointer();
        }
    }

private:
    FloatVectorReader *_holderImpl;
    size_t _dimension;
    VectorHolder::FeatureType _type;
    const bool _catEnabled = false;
};

} // namespace mercury

#endif // __FILE_INDEX_HOLDER_H__
