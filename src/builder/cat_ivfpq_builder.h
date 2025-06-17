#pragma once

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
#include <unordered_map>
#include <unordered_set>

namespace mercury {

class QueryDistanceMatrix;

class CatIvfpqBuilder : public IndexBuilder
{
public:
    //! Constructor
    CatIvfpqBuilder() :_globalId(0),
                    _segment(0),
                    _isTrainDone(true),
                    _isRuned(false)
    {}

    //! Destructor
    ~CatIvfpqBuilder() override;

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
    bool IsFinish();
    std::list<Closure::Pointer> JobSplit(const VectorHolder::Pointer &holder);

private:
    bool InitResource(PQCodebook::Pointer pqCodebook);
    bool InitResource(const std::string& rough_matrix, const std::string& integrate_matrix);

    int initProfile(size_t elemCount, size_t elemSize);
    size_t memQuota2DocCount(size_t memQuota, size_t elemSize);
    void singleTaskWithoutLabels(cat_t cat_, uint64_t key, std::shared_ptr<char> data);
    bool doSingleBuild(cat_t cat_, uint64_t key,
                       std::shared_ptr<char> data,
                       int32_t roughLabel,
                       const std::vector<uint16_t>& productLabels);

    bool flushSegment();
    bool flushWithAdjust();
    int64_t loadSegments(std::vector<std::unique_ptr<mercury::MMapFile>> &fileHolder,
                         std::vector<CoarseIndex::Pointer> &indexSegs,
                         std::vector<ArrayProfile::Pointer> &pkSegs,
                         std::vector<ArrayProfile::Pointer> &catSegs,
                         std::vector<ArrayProfile::Pointer> &productSegs,
                         std::vector<ArrayProfile::Pointer> &featureSegs);
    bool writeIndexPackage(size_t maxDocNum,
                           const std::string &prefix,
                           const mercury::IndexStorage::Pointer &stg,
                           const mercury::MMapFile &indexMergeFile,
                           const mercury::MMapFile &pkMergeFile,
                           const mercury::MMapFile &keyCatMapMergeFile,
                           const mercury::MMapFile &catSetMergeFile,
                           const mercury::MMapFile &productMergeFile,
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

    CentroidResource::Pointer _resource;
    CoarseIndex _coarseIndex;
    ArrayProfile _pkProfile;
    ArrayProfile _catProfile;
    ArrayProfile _pqcodeProfile;
    ArrayProfile _featureProfile;
    std::shared_ptr<char> _coarseBase;
    std::shared_ptr<char> _pkBase;
    std::shared_ptr<char> _catBase;
    std::shared_ptr<char> _productBase;
    std::shared_ptr<char> _featureBase;
    bool _isTrainDone;
    bool _isRuned;

    mercury::MMapFile roughFile, integrateFile;
    std::string rough_matrix;
    std::string integrate_matrix;
};

} // namespace mercury
