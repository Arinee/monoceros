#include "local_file_writer.h"
#include "src/core/common/common_define.h"

namespace mercury
{

LocalFileWriter::LocalFileWriter(const std::string &file)
    : _file(file)
{
}

LocalFileWriter::~LocalFileWriter(void)
{
    if (_fp != nullptr) {
        fclose(_fp);
    }
    _fp = nullptr;
}

bool LocalFileWriter::init(void)
{
    _fp = fopen(_file.c_str(), "wb");
    if (_fp == nullptr) {
        LOG_ERROR("Open file [%s] failed", _file.c_str());
        return false;
    }
    
    return true;
}

size_t LocalFileWriter::write(const void *buf, size_t len)
{
    size_t writeSize = fwrite(buf, 1, len, _fp);
    if (writeSize != len) {
        LOG_ERROR("Write file [%s] failed", _file.c_str());
        return static_cast<size_t>(-1);
    }

    return writeSize;
}

size_t LocalFileWriter::write(size_t offset, const void *buf, size_t len)
{
    int iRet = seek(static_cast<long>(offset), SEEK_SET);
    if (iRet != 0) {
        LOG_ERROR("Seek file failed");
        return static_cast<size_t>(-1);
    }

    size_t writeSize = fwrite(buf, 1, len, _fp);
    if (writeSize != len) {
        LOG_ERROR("Write file [%s] failed", _file.c_str());
        return static_cast<size_t>(-1);
    }

    return writeSize;
}

int LocalFileWriter::seek(long offset, int fromWhere)
{
    if (_fp == nullptr) {
        LOG_ERROR("Seek using unopened fp");
        return mercury::IndexError_Uninitialized;
    }

    return fseek(_fp, offset, fromWhere);
}

int LocalFileWriter::truncate(size_t len)
{
    int iRet = ftruncate(fileno(_fp), len);
    if (iRet != 0) {
        LOG_ERROR("Ftruncate file failed");
        return mercury::IndexError_TruncateFile;
    }

    return 0;
}

void LocalFileWriter::close(void)
{
    fclose(_fp);
}

} //mercury
