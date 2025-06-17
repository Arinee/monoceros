#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include "framework/index_logger.h"
#include "array_profile.h"

using namespace mercury;

ArrayProfile::ArrayProfile() 
    : _base(nullptr)
    , _header(nullptr)
    , _infos(nullptr)
{
    pthread_spin_init(&_spinLock, PTHREAD_PROCESS_PRIVATE);
}

ArrayProfile::~ArrayProfile()
{
    pthread_spin_destroy(&_spinLock);
}

bool ArrayProfile::create(void *pBase, size_t len, int64_t infoSize)
{
    assert(pBase != nullptr);
    assert(infoSize > 0);
    int64_t maxDocSize = (len - sizeof(Header)) / infoSize;
    size_t capacity = sizeof(Header) + infoSize * maxDocSize;
    if (capacity > len) {
        return false;
    }

    _base = (char *)pBase;

    _header = (Header *) _base;

    _header->usedDocNum = 0; // NOTE: 已使用docnum为0
    _header->maxDocNum = maxDocSize;
    _header->capacity = capacity;
    _header->infoSize = infoSize;
    memset(_header->padding, 0, sizeof(_header->padding));

    _infos = _base + sizeof(Header);
    return true;
}

bool ArrayProfile::load(void *pBase, size_t len) 
{
    _base = (char *)pBase;
    _header = (Header *) _base;
    if ((size_t)_header->capacity != len) {
        return false;
    }
    _infos = _base + sizeof(Header);
    return true;
}

bool ArrayProfile::unload() 
{
    return true;
}

void ArrayProfile::reset()
{
    //pthread_spin_lock(&_spinLock);
    _header->usedDocNum = 0;
    //pthread_spin_unlock(&_spinLock);
}

size_t ArrayProfile::CalcSize(size_t docNum, size_t docSize)
{
    return sizeof(Header) + docSize * docNum;
}

bool ArrayProfile::dump(const std::string &file)
{
    FILE *fp = fopen(file.c_str(), "wb");
    if (!fp) {
        LOG_ERROR("Fopen file [%s] with wb failed:[%s]", file.c_str(), strerror(errno));
        return false;
    }

    // header should be written in the end, to ensure integrity
    off_t headSize = sizeof(*_header);
    int ret = fseek(fp, headSize, SEEK_SET);
    if (ret != 0) {
        LOG_ERROR("Seek file [%s] failed:[%s]", file.c_str(), strerror(errno));
        fclose(fp);
        return false;
    }
    
    const off_t SIZE_ONE_TIME = 100 * 1024 * 1024;
    off_t leftSize = _header->capacity - headSize;
    char *curPos = reinterpret_cast<char *>(_header) + headSize;
    while (leftSize > 0) {
        off_t curWriteSize = (leftSize < SIZE_ONE_TIME ? leftSize : SIZE_ONE_TIME);
        off_t writeSize = fwrite(curPos, 1, curWriteSize, fp);
        if (writeSize != curWriteSize) {
            LOG_ERROR("Write to file [%s] failed:[%s], file size:%ld, left size:%ld", 
                      file.c_str(), strerror(errno), _header->capacity, leftSize);
            fclose(fp);
            return false;
        }

        curPos += writeSize;
        leftSize -= writeSize;
    }
    rewind(fp);
    off_t cnt = fwrite(_header, headSize, 1, fp);
    if (cnt != 1) {
        LOG_ERROR("Write file header to file [%s] fail", file.c_str());
        fclose(fp);
        return false;
    }
    fclose(fp);
    return true;
}

void ArrayProfile::serialize(std::string &output) const
{
    output = std::string(_base, _header->capacity);
}
