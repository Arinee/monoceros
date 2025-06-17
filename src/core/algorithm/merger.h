#ifndef __MERCURY_CORE_MERGER_H__
#define __MERCURY_CORE_MERGER_H__

#include <vector>
#include <string>
#include "index.h"

MERCURY_NAMESPACE_BEGIN(core);
typedef std::vector<std::pair<size_t, docid_t>> MergeIdMap; //new doc_id --> original index + docid
typedef std::vector<std::pair<size_t, pk_t>> MergePkMap; //new doc_id --> original index + PK

/*! Index Merger
 */
class Merger
{
public:
    //! Index Merger Pointer
    typedef std::shared_ptr<Merger> Pointer;

    //! Destructor
    virtual ~Merger() {}

    //! Initialize Merger
    virtual int Init(const IndexParams &params) = 0;

    virtual int MergeByDocid(const std::vector<Index::Pointer> &indexes, const MergeIdMap& idMap) = 0;

    virtual int MergeByPk(const std::vector<Index::Pointer> &indexes, const MergePkMap& idMap) = 0;

    //! Dump index into file or memory
    virtual int DumpIndex(const std::string &path, const IndexStorage::Pointer &stg) = 0;

    virtual const void * DumpIndex(size_t *size) = 0;

    virtual const void *DumpCentroid(size_t *size) {
        *size = 0;
        return nullptr;
    }
};

MERCURY_NAMESPACE_END(core);

#endif // __MERCURY_CORE_MERGER_H__
