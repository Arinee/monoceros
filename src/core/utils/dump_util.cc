#include "dump_util.h"
#include "src/core/common/common_define.h"

MERCURY_NAMESPACE_BEGIN(core);

int DumpUtil::dump(char *data, uint64_t len, const std::string &file)
{
    FILE *fp = fopen(file.c_str(), "wb");
    if (NULL == fp) {
        LOG_ERROR("Fopen file [%s] with wb failed:[%s]", file.c_str(), strerror(errno));
        return false;
    }

    size_t leftSize = len;
    char *curPos = data;
    while (leftSize > 0) {
        size_t curWriteSize = (leftSize < SIZE_WRITE_ONE_TIME) ? leftSize : SIZE_WRITE_ONE_TIME;
        size_t writeSize = fwrite(curPos, 1, curWriteSize, fp);
        if (writeSize != curWriteSize) {
            LOG_ERROR("Write data to file [%s] failed", file.c_str());
            return mercury::IndexError_WriteFile;
        }

        curPos += writeSize;
        leftSize -= writeSize;
    }

    fclose(fp);
    return 0;
    
}

int DumpUtil::dump(char *data, uint64_t len,
                   const std::string &file,
                   const mercury::core::IndexStorage::Pointer &stg)
{
    auto handler = stg->create(file, len);
    if (!handler) {
        LOG_ERROR("Storage create handler  failed");
        return mercury::IndexError_CreateStorageHandler;
    }

    size_t leftSize = len;
    char *curPos = data;
    while (leftSize > 0) {
        size_t curWriteSize = (leftSize < SIZE_WRITE_ONE_TIME) ? leftSize : SIZE_WRITE_ONE_TIME;
        size_t writeSize = handler->write(curPos, curWriteSize);
        if (writeSize != curWriteSize) {
            LOG_ERROR("Write handler failed");
            return mercury::IndexError_WriteStorageHandler;
        }

        curPos += writeSize;
        leftSize -= writeSize;
    }

    return 0;
}

MERCURY_NAMESPACE_END(core);
