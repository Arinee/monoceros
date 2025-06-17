/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     index_error.h
 *   \author   qiuming
 *   \date     Jan 2019
 *   \version  1.0.0
 *   \brief    index error
 */

#ifndef __MERCURY_INDEX_ERROR_H__
#define __MERCURY_INDEX_ERROR_H__

#include <map>

namespace mercury
{

/*! Index Error
 */
class IndexError
{
public:
    /*! Index Error Code
     */
    class Code
    {
    public:
        //! Constructor
        Code(int val, const char *str) : _value(-val), _desc(str)
        {
            IndexError::Instance()->emplace(this);
        }

        //! Retrieve the value of code
        operator int() const
        {
            return (this->_value);
        }

        //! Retrieve the value of code
        int value() const
        {
            return (this->_value);
        }

        //! Retrieve the description of code
        const char *desc() const
        {
            return (this->_desc);
        }

    private:
        int _value;
        const char *_desc;
    };

    //! Retrieve the description of code
    static const char *What(int val)
    {
        return IndexError::Instance()->what(val);
    }

protected:
    //! Constructor
    IndexError(void) : _map() {}

    //! Inserts a new code into map
    void emplace(const IndexError::Code *code)
    {
        _map.emplace(code->value(), code);
    }

    //! Retrieve the description of code
    const char *what(int val) const
    {
        auto iter = _map.find(val);
        if (iter != _map.end()) {
            return iter->second->desc();
        }
        return "";
    }

    //! Retrieve the singleton
    static IndexError *Instance(void)
    {
        static IndexError error;
        return (&error);
    }

private:
    //! Disable them
    IndexError(const IndexError &) = delete;
    IndexError(IndexError &&) = delete;
    IndexError &operator=(const IndexError &) = delete;

    //! Error code map
    std::map<int, const IndexError::Code *> _map;
};

//! Index Error Code Define
#define INDEX_ERROR_CODE_DEFINE(__NAME__, __VAL__, __DESC__)                   	\
    const mercury::IndexError::Code IndexError_##__NAME__((__VAL__),           	\
                                                          (__DESC__));         	\
    const mercury::IndexError::Code &_IndexErrorCode_##__VAL__##_Register(     	\
        IndexError_##__NAME__)

//! Index Error Code Declare
#define INDEX_ERROR_CODE_DECLARE(__NAME__)                                     	\
    extern const mercury::IndexError::Code IndexError_##__NAME__

//! Build-in error code
INDEX_ERROR_CODE_DECLARE(Runtime); // Runtime error
INDEX_ERROR_CODE_DECLARE(Logic);   // Logic error
INDEX_ERROR_CODE_DECLARE(Type);    // Type error
INDEX_ERROR_CODE_DECLARE(System);  // System call error
INDEX_ERROR_CODE_DECLARE(Cast);    // Cast error
INDEX_ERROR_CODE_DECLARE(IO);      // IO error

INDEX_ERROR_CODE_DECLARE(InvalidArgument); // Invalid argument
INDEX_ERROR_CODE_DECLARE(NotImplemented);  // Not implemented
INDEX_ERROR_CODE_DECLARE(Unsupported);     // Unsupported
INDEX_ERROR_CODE_DECLARE(Denied);          // Permission denied
INDEX_ERROR_CODE_DECLARE(Canceled);        // Operation canceled

INDEX_ERROR_CODE_DECLARE(Overflow);      // Overflow
INDEX_ERROR_CODE_DECLARE(Underflow);     // Underflow
INDEX_ERROR_CODE_DECLARE(InvalidLength); // Invalid length
INDEX_ERROR_CODE_DECLARE(OutOfRange);    // Out of range
INDEX_ERROR_CODE_DECLARE(NoBuffer);      // No buffer space available
INDEX_ERROR_CODE_DECLARE(NoMemory);      // Not enough space
INDEX_ERROR_CODE_DECLARE(NoParamFound);  // No parameter found
INDEX_ERROR_CODE_DECLARE(InvalidValue);  // Invalid value
INDEX_ERROR_CODE_DECLARE(Mismatch);      // Mismatch
INDEX_ERROR_CODE_DECLARE(Uninitialized); // Uninitialized
INDEX_ERROR_CODE_DECLARE(IndexFull);     // IndexFull
INDEX_ERROR_CODE_DECLARE(Exist);         // Already exist

// Fail to create a storage handler
INDEX_ERROR_CODE_DECLARE(CreateStorageHandler);
// Fail to open a storage handler
INDEX_ERROR_CODE_DECLARE(OpenStorageHandler);
// Fail to read data from a storage handler
INDEX_ERROR_CODE_DECLARE(ReadStorageHandler);
// Fail to write data into a storage handler
INDEX_ERROR_CODE_DECLARE(WriteStorageHandler);

INDEX_ERROR_CODE_DECLARE(Serialize);   // Serialize error
INDEX_ERROR_CODE_DECLARE(Deserialize); // Deserialize error

INDEX_ERROR_CODE_DECLARE(UnmatchedMeta);      // Unmatched index meta
INDEX_ERROR_CODE_DECLARE(IndexLoaded);        // Index loaded
INDEX_ERROR_CODE_DECLARE(NoIndexLoaded);      // No index loaded
INDEX_ERROR_CODE_DECLARE(LoadPackageIndex);   // Fail to load package index
INDEX_ERROR_CODE_DECLARE(LoadIndexMeta);      // Fail to load index meta
INDEX_ERROR_CODE_DECLARE(DumpPackageIndex);   // Fail to dump package index
INDEX_ERROR_CODE_DECLARE(InvalidFeature);     // Invalid feature
INDEX_ERROR_CODE_DECLARE(InvalidKey);         // Invalid key
INDEX_ERROR_CODE_DECLARE(InvalidFeatureSize); // Invalid feature size
INDEX_ERROR_CODE_DECLARE(InvalidKeySize);     // Invalid key size
INDEX_ERROR_CODE_DECLARE(NoMetaFound);        // No meta found
INDEX_ERROR_CODE_DECLARE(NoFeatureFound);     // No feature found
INDEX_ERROR_CODE_DECLARE(NoKeyFound);         // No key found
INDEX_ERROR_CODE_DECLARE(EmptyFile);          // Empty file
INDEX_ERROR_CODE_DECLARE(InsertHashFailed);   // Insert hash failed
INDEX_ERROR_CODE_DECLARE(InvalidIterator);    // Invalid iterator failed
INDEX_ERROR_CODE_DECLARE(RemoveHashFailed);   // Remove key from hash failed

INDEX_ERROR_CODE_DECLARE(CreateFile);   // Create file error
INDEX_ERROR_CODE_DECLARE(OpenFile);     // Open file error
INDEX_ERROR_CODE_DECLARE(WriteFile);    // Write file error
INDEX_ERROR_CODE_DECLARE(SeekFile);     // Seek file error
INDEX_ERROR_CODE_DECLARE(CloseFile);    // Close file error
INDEX_ERROR_CODE_DECLARE(TruncateFile); // Trucate file error
INDEX_ERROR_CODE_DECLARE(MMapFile);     // MMap file error

INDEX_ERROR_CODE_DECLARE(CreateDirectory); // Create dir error
INDEX_ERROR_CODE_DECLARE(OpenDirectory);   // Open dir error

INDEX_ERROR_CODE_DECLARE(CreateIterator); // Create iterator

INDEX_ERROR_CODE_DECLARE(Clustering); // Clustering error
INDEX_ERROR_CODE_DECLARE(NotTrained); // Not trained

} // namespace mercury

#endif // __MERCURY_INDEX_ERROR_H__

