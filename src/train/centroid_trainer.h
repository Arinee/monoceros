#ifndef MERCURY_CENTROID_TRAINER_BUILDER_H__
#define MERCURY_CENTROID_TRAINER_BUILDER_H__

#include <mutex>
#include <atomic>
#include <vector>
#include "index/centroid_resource.h"
#include "builder/multthread_batch_workflow.h"
#include "common/pq_codebook.h"
#include "common/params_define.h"
#include "framework/index_framework.h"
#include "framework/vector_holder.h"

namespace mercury {

class CentroidTrainer
{
public:
    //! Constructor
    CentroidTrainer(bool roughOnly_ = false, bool sanityCheck_ = false,
                    double sanityCheckCentroidNumRatio_ = 0.5):
        _isTrainDone(false), _isRuned(false),
        _roughOnly(roughOnly_), _sanityCheck(sanityCheck_),
        _sanityCheckCentroidNumRatio(sanityCheckCentroidNumRatio_) {}

    //! Initialize Data Clean & Dump
    bool Init(const mercury::IndexMeta &meta, const mercury::IndexParams &params);
    int trainIndexImpl(const VectorHolder::Pointer &holder);

private:
    int trainRoughAndIntegrate(const VectorHolder::Pointer &holder);
    int trainRough(const VectorHolder::Pointer &holder);


private:
    mercury::IndexMeta _meta;
    mercury::IndexParams _params;
    std::shared_ptr<mercury::ThreadPool> _pool;
    CentroidResource::Pointer _resource;
    bool _isTrainDone;
    bool _isRuned;
    bool _roughOnly;
    bool _sanityCheck;
    double _sanityCheckCentroidNumRatio;
};

} // namespace mercury

#endif // MERCURY_CENTROID_TRAINER_BUILDER_H__
