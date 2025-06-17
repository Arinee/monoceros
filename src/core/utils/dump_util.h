#ifndef DUMP_UTIL_H_
#define DUMP_UTIL_H_

#include "src/core/framework/index_framework.h"
#include <stdio.h>
#include "src/core/common/common_define.h"

MERCURY_NAMESPACE_BEGIN(core);
class DumpUtil
{
public:
    static int dump(char *data, uint64_t len, const std::string &file);
    static int dump(char *data, uint64_t len, 
                    const std::string &file,
                    const mercury::core::IndexStorage::Pointer &stg);
    template <typename T>
    static int pad(T &handler, uint64_t len);
};

template <typename T>
int DumpUtil::pad(T &handler, uint64_t len)
{
    //create padding buffer
    const uint64_t bufSize = 1024 * 1024;
    char *buf = new (std::nothrow) char[bufSize];
    if (buf == nullptr) {
        LOG_ERROR("New memory for buf failed");
        return mercury::IndexError_NoMemory;
    }
    memset(buf, 0, bufSize);
    std::unique_ptr<char[]> bufPtr(buf);

    //padding
    uint64_t leftSize = len;
    while (leftSize > 0) {
        size_t curWriteSize = (leftSize < bufSize) ? leftSize : bufSize;
        size_t writeSize = handler->write(buf, curWriteSize);
        if (writeSize != curWriteSize) {
            LOG_ERROR("Write data to handler failed");
            return mercury::IndexError_WriteStorageHandler;
        }

        leftSize -= writeSize;
    }
    
    return 0;
}

MERCURY_NAMESPACE_END(core);
#endif //DUMP_UTIL_H_
