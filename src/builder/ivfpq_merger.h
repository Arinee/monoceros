#ifndef MERCURY_BUILDER_IVFPQ_MERGER_H_
#define MERCURY_BUILDER_IVFPQ_MERGER_H_

#include <mutex>
#include <atomic>
#include <vector>
#include <list>
#include "framework/index_merger.h"
#include "index/index_ivfpq.h"
#include "index/coarse_index.h"
#include "index/array_profile.h"
#include "index/centroid_resource.h"
#include "common/params_define.h"
#include "framework/index_framework.h"
#include "framework/utility/mmap_file.h"

namespace mercury {

class QueryDistanceMatrix;

class IvfpqMerger : public IndexMerger
{
public:
    //! Constructor
    IvfpqMerger() = default;

    //! Destructor
    ~IvfpqMerger() override = default;

    //! Initialize Merger
    int Init(const IndexParams &params) override;

    //! Cleanup Merger
    int Cleanup() override;

    //! Feed indexes from file paths or dirs
    int FeedIndex(const std::vector<std::string> &paths,
                  const IndexStorage::Pointer &stg) override;

    //! Merge operator
    int MergeIndex() override;

    //! Dump index into file path or dir
    int DumpIndex(const std::string &prefix,
                  const IndexStorage::Pointer &stg) override;

private:
    static const size_t MIN_BUILD_COUNT = 100000;

    IndexMeta _meta;
    std::string _segmentDir;

    CentroidResource* _resource = nullptr;
    std::vector<IndexIvfpq::Pointer> _indexes;
    size_t _mergedDocNum = 0;

    MMapFile indexMergeFile;
    MMapFile pkMergeFile;
    MMapFile productMergeFile;
    MMapFile featureMergeFile;
    MMapFile idMapMergeFile;

    CoarseIndex::Pointer coarseIndex;
    ArrayProfile::Pointer pkProfile;
    ArrayProfile::Pointer pqcodeProfile;
    ArrayProfile::Pointer featureProfile;
    HashTable<uint64_t, docid_t>::Pointer idMapPtr;

};

} // namespace mercury

#endif // MERCURY_BUILDER_IVFPQ_MERGER_H_
