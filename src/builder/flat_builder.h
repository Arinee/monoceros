#ifndef MERCURY_FLAT_BUILDER_H__
#define MERCURY_FLAT_BUILDER_H__

#include <mutex>
#include <atomic>
#include <vector>
#include <list>
#include "index/coarse_index.h"
#include "index/array_profile.h"
#include "index/centroid_resource.h"
#include "multthread_batch_workflow.h"
#include "common/pq_codebook.h"
#include "common/params_define.h"
#include "framework/index_builder.h"
#include "framework/index_framework.h"
#include "framework/vector_holder.h"
#include "framework/utility/mmap_file.h"
#include "framework/utility/thread_pool.h"

namespace mercury {

class FlatBuilder : public IndexBuilder
{
public:
    //! Constructor
    FlatBuilder() : _globalId(0), _segment(0) {}

    //! Destructor
    ~FlatBuilder() override;

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
    void singleTask(uint64_t key, std::shared_ptr<char> data);
    bool flushWithAdjust();
    int64_t loadSegments(std::vector<std::unique_ptr<mercury::MMapFile>> &fileHolder,
            std::vector<ArrayProfile::Pointer> &pkSegs,
            std::vector<ArrayProfile::Pointer> &featureSegs);
    bool writeIndexPackage(size_t maxDocNum,
        const std::string &prefix,
        const IndexStorage::Pointer &stg,
        const mercury::MMapFile &pkMergeFile,
        const mercury::MMapFile &featureMergeFile,
        const mercury::MMapFile &idMapMergeFile);

private:
    static const size_t MIN_BUILD_COUNT = 100000;

    std::mutex _docidLock;
    mercury::IndexMeta _meta;
    mercury::IndexParams _params;
    std::atomic<int64_t> _globalId;
    size_t _segment;
    std::string _segmentDir;
    std::vector<std::string> _segmentList;

    ArrayProfile _pkProfile;
    ArrayProfile _featureProfile;
    std::shared_ptr<char> _pkBase;
    std::shared_ptr<char> _featureBase;
};

} // namespace mercury

#endif // MERCURY_FLAT_BUILDER_H__
