#ifndef __REDINDEX_HDFS_FILE_WRAPPER_H
#define __REDINDEX_HDFS_FILE_WRAPPER_H

#include "fslib/fslib.h"
#include "fslib/fs/FileSystem.h"
#include "fslib/fs/File.h"

namespace mercury {
namespace core {

using namespace std;
using namespace fslib;
using namespace fslib::fs;

class HdfsFileWrapper {
 public:
  static bool IsExistIgnoreError(const string& filePath)
  {
    ErrorCode ec = FileSystem::isExist(filePath);
    if (ec == EC_TRUE)
      {
        return true;
      }
    else if (ec == EC_FALSE)
      {
        return false;
      }

    std::cerr<<"Determine existence of file "<<filePath<<std::endl;
    return false;
  }

  inline static std::string GetErrorString(fslib::ErrorCode ec)
  {
    return fslib::fs::FileSystem::getErrorString(ec);
  }

  static bool AtomicLoad(const std::string& filePath,
                         std::string& content) {
    if (!IsExistIgnoreError(filePath))
      {
        std::cerr<<"file not exist:"<<filePath<<std::endl;
      }

    FileMeta fileMeta;
    ErrorCode ret = FileSystem::getFileMeta(filePath, fileMeta);
    if (ret != EC_OK)
      {
        std::cerr<<"Get file meta of "<<filePath<<"failed. error:"<<GetErrorString(ret)<<std::endl;
        return false;
      }
    FilePtr file(FileSystem::openFile(filePath, READ));
    if (!file->isOpened())
      {
        std::cerr<<"open file "<<filePath<<"failed. error:"<<GetErrorString(ret)<<std::endl;
        return false;
      }
    size_t fileLength = fileMeta.fileLength;
    content.resize(fileLength);
    char* data = const_cast<char*>(content.c_str());

    ssize_t readBytes = file->read(data, fileLength);
    if (readBytes != (ssize_t)fileLength)
      {
        std::cerr<<"read file "<<filePath<<"failed. error:"<<GetErrorString(ret)<<std::endl;
        return false;
      }

    ret = file->close();
    if (ret != EC_OK)
      {
        std::cerr<<"close file "<<filePath<<"failed. error:"<<GetErrorString(ret)<<std::endl;
        return false;
      }
    return true;

  }
};

} // namespace core
} // namespace mercury

#endif // __REDINDEX_HDFS_FILE_WRAPPER_H
