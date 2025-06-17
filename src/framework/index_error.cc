/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     index_error.cc
 *   \author   qiuming
 *   \date     Jan 2019
 *   \version  1.0.0
 *   \brief    Implementation of Index Error
 */

#include "index_error.h"

namespace mercury 
{

INDEX_ERROR_CODE_DEFINE(Runtime, 1, "Runtime error");
INDEX_ERROR_CODE_DEFINE(Logic, 2, "Logic error");
INDEX_ERROR_CODE_DEFINE(Type, 3, "Type error");
INDEX_ERROR_CODE_DEFINE(System, 4, "System call error");
INDEX_ERROR_CODE_DEFINE(Cast, 5, "Cast error");
INDEX_ERROR_CODE_DEFINE(IO, 6, "IO error");

INDEX_ERROR_CODE_DEFINE(InvalidArgument, 10, "Invalid argument");
INDEX_ERROR_CODE_DEFINE(NotImplemented, 11, "Not implemented");
INDEX_ERROR_CODE_DEFINE(Unsupported, 12, "Unsupported");
INDEX_ERROR_CODE_DEFINE(Denied, 13, "Permission denied");
INDEX_ERROR_CODE_DEFINE(Canceled, 14, "Operation canceled");

INDEX_ERROR_CODE_DEFINE(Overflow, 15, "Overflow");
INDEX_ERROR_CODE_DEFINE(Underflow, 16, "Underflow");
INDEX_ERROR_CODE_DEFINE(InvalidLength, 17, "Invalid length");
INDEX_ERROR_CODE_DEFINE(OutOfRange, 18, "Out of range");
INDEX_ERROR_CODE_DEFINE(NoBuffer, 19, "No buffer space available");
INDEX_ERROR_CODE_DEFINE(NoMemory, 20, "Not enough space");
INDEX_ERROR_CODE_DEFINE(NoParamFound, 21, "No parameter found");
INDEX_ERROR_CODE_DEFINE(InvalidValue, 22, "Invalid value");
INDEX_ERROR_CODE_DEFINE(Mismatch, 23, "Mismatch");
INDEX_ERROR_CODE_DEFINE(Uninitialized, 24, "Uninitialized");
INDEX_ERROR_CODE_DEFINE(IndexFull, 25, "Index is full");
INDEX_ERROR_CODE_DEFINE(Exist, 26, "Already exist");

INDEX_ERROR_CODE_DEFINE(CreateStorageHandler, 101,
                        "Fail to create a storage handler");
INDEX_ERROR_CODE_DEFINE(OpenStorageHandler, 102,
                        "Fail to open a storage handler");
INDEX_ERROR_CODE_DEFINE(ReadStorageHandler, 103,
                        "Fail to read data from a storage handler");
INDEX_ERROR_CODE_DEFINE(WriteStorageHandler, 104,
                        "Fail to write data into a storage handler");
INDEX_ERROR_CODE_DEFINE(Serialize, 105, "Serialize error");
INDEX_ERROR_CODE_DEFINE(Deserialize, 106, "Deserialize error");

INDEX_ERROR_CODE_DEFINE(UnmatchedMeta, 110, "Unmatched index meta");
INDEX_ERROR_CODE_DEFINE(IndexLoaded, 111, "Index loaded");
INDEX_ERROR_CODE_DEFINE(NoIndexLoaded, 112, "No index loaded");
INDEX_ERROR_CODE_DEFINE(LoadPackageIndex, 113, "Fail to load package index");
INDEX_ERROR_CODE_DEFINE(LoadIndexMeta, 114, "Fail to load index meta");
INDEX_ERROR_CODE_DEFINE(DumpPackageIndex, 115, "Fail to dump package index");
INDEX_ERROR_CODE_DEFINE(InvalidFeature, 116, "Invalid feature");
INDEX_ERROR_CODE_DEFINE(InvalidKey, 117, "Invalid key");
INDEX_ERROR_CODE_DEFINE(InvalidFeatureSize, 118, "Invalid feature size");
INDEX_ERROR_CODE_DEFINE(InvalidKeySize, 119, "Invalid key size");
INDEX_ERROR_CODE_DEFINE(NoMetaFound, 120, "No meta found");
INDEX_ERROR_CODE_DEFINE(NoFeatureFound, 121, "No feature found");
INDEX_ERROR_CODE_DEFINE(NoKeyFound, 122, "No key found");
INDEX_ERROR_CODE_DEFINE(EmptyFile, 123, "Empty file");
INDEX_ERROR_CODE_DEFINE(InsertHashFailed, 124, "Insert hash failed");
INDEX_ERROR_CODE_DEFINE(InvalidIterator, 125, "Invalid iterator");
INDEX_ERROR_CODE_DEFINE(RemoveHashFailed, 126, "Remove key from hash failed");

INDEX_ERROR_CODE_DEFINE(CreateFile, 150, "Create file error");
INDEX_ERROR_CODE_DEFINE(OpenFile, 151, "Open file error");
INDEX_ERROR_CODE_DEFINE(WriteFile, 152, "Write file error");
INDEX_ERROR_CODE_DEFINE(SeekFile, 153, "Seek file error");
INDEX_ERROR_CODE_DEFINE(CloseFile, 154, "Close file error");
INDEX_ERROR_CODE_DEFINE(TruncateFile, 155, "TruncateFile file error");
INDEX_ERROR_CODE_DEFINE(MMapFile, 156, "MMap file error");

INDEX_ERROR_CODE_DEFINE(CreateDirectory, 160, "Create directory error");
INDEX_ERROR_CODE_DEFINE(OpenDirectory, 161, "Open directory error");

INDEX_ERROR_CODE_DEFINE(CreateIterator, 170, "Create iterator error");

INDEX_ERROR_CODE_DEFINE(Clustering, 180, "Clustering error");
INDEX_ERROR_CODE_DEFINE(NotTrained, 181, "Not trained");

} // namespace mercury

