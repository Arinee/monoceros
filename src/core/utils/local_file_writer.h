#ifndef MERCURY_LOCAL_FILE_WRITER_H_
#define MERCURY_LOCAL_FILE_WRITER_H_

#include <stdio.h>
#include <memory>
#include <string>
#include "src/core/framework/index_framework.h"

namespace mercury
{

//uniform write interface with Handler to let tempalte work
class LocalFileWriter
{
public:
    typedef std::shared_ptr<LocalFileWriter> Pointer;

public:
    LocalFileWriter(const std::string &file);
    ~LocalFileWriter(void);

public:
    bool init(void);
    size_t write(const void *buf, size_t len);
    size_t write(size_t offset, const void *buf, size_t len);
    int seek(long offset, int fromWhere);
    int truncate(size_t len);
    void close(void);

private:
    std::string _file;
    FILE *_fp;
};

} //mercury
#endif //MERCURY_LOCAL_FILE_WRITER_H_
