#ifndef MERCURY_CAT_FLAT_BUILDER_H__
#define MERCURY_CAT_FLAT_BUILDER_H__

#include <list>
#include "flat_builder.h"
#include "utils/hash_table.h"
#include "index/coarse_index.h"

namespace mercury {

class CatFlatBuilder : public IndexBuilder
{
public:
    //! Constructor
    CatFlatBuilder() : _globalId(0), _segment(0) {}

    //! Destructor
    ~CatFlatBuilder() override;

    //! Initialize Builder
    int Init(const IndexMeta &meta, const IndexParams &params) override;

    //! Cleanup Builder
    int Cleanup() override;

    //! Train the data
    int Train(const VectorHolder::Pointer &holder) override;

    //! Build the index
    int BuildIndex(const VectorHolder::Pointer &holder) override;

    //! Dump index into file or memory
    int DumpIndex(const std::string &path, const IndexStorage::Pointer &stg) override;

    //! Job
    std::list<Closure::Pointer> JobSplit(const VectorHolder::Pointer &holder);

private:
    int initProfile(size_t elemCount, size_t elemSize);
    size_t memQuota2DocCount(size_t memQuota, size_t elemSize);
    void singleTask(cat_t cat_, uint64_t key, std::shared_ptr<char> data);
    bool flushWithAdjust();
    int64_t loadSegments(std::vector<std::unique_ptr<mercury::MMapFile>> &fileHolder,
            std::vector<ArrayProfile::Pointer> &pkSegs,
            std::vector<ArrayProfile::Pointer> &catSegs,
            std::vector<ArrayProfile::Pointer> &featureSegs);
    bool writeIndexPackage(size_t maxDocNum,
        const std::string &prefix,
        const IndexStorage::Pointer &stg,
        const mercury::MMapFile &pkMergeFile,
        const mercury::MMapFile &featureMergeFile,
        const mercury::MMapFile &idMapMergeFile,
        const mercury::MMapFile &catSlotMergeFile,
        const mercury::MMapFile &slotDocMergeFile);

private:
    static const size_t MIN_BUILD_COUNT = 100000;

    std::mutex _docidLock;
    mercury::IndexMeta _meta;
    mercury::IndexParams _params;
    std::atomic<int64_t> _globalId;
    std::atomic<slot_t> _globalSlot;
    size_t _segment;
    std::string _segmentDir;
    std::vector<std::string> _segmentList;

    ArrayProfile _pkProfile;
    ArrayProfile _catProfile;
    ArrayProfile _featureProfile;
    HashTable<cat_t, slot_t> _catSlotMap;
    CoarseIndex _slotDocIndex;
    std::shared_ptr<char> _pkBase;
    std::shared_ptr<char> _catBase;
    std::shared_ptr<char> _featureBase;
};

} // namespace mercury

#endif // MERCURY_CAT_FLAT_BUILDER_H__
