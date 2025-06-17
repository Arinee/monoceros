#include "algorithm_factory.h"
#include "gpu_group_ivf/gpu_group_ivf_searcher.h"
#include "gpu_group_ivf_pq/gpu_group_ivf_pq_searcher.h"
#include "gpu_group_ivf_pq_small/gpu_group_ivf_pq_small_searcher.h"
#include "gpu_ivf_rpq/gpu_ivf_rpq_searcher.h"
#include "group_hnsw/group_hnsw_builder.h"
#include "group_hnsw/group_hnsw_merger.h"
#include "group_hnsw/group_hnsw_searcher.h"
#include "group_ivf/group_ivf_builder.h"
#include "group_ivf/group_ivf_merger.h"
#include "group_ivf/group_ivf_searcher.h"
#include "group_ivf_pq/group_ivf_pq_builder.h"
#include "group_ivf_pq/group_ivf_pq_merger.h"
#include "group_ivf_pq/group_ivf_pq_searcher.h"
#include "group_ivf_pq_small/group_ivf_pq_small_builder.h"
#include "group_ivf_pq_small/group_ivf_pq_small_merger.h"
#include "group_ivf_pq_small/group_ivf_pq_small_searcher.h"
#include "vamana/disk_vamana_index.h"
#include "vamana/disk_vamana_builder.h"
#include "vamana/disk_vamana_searcher.h"
#include "vamana/ram_vamana_index.h"
#include "vamana/ram_vamana_builder.h"
#include "vamana/ram_vamana_searcher.h"
#include "ivf/ivf_builder.h"
#include "ivf/ivf_merger.h"
#include "ivf/ivf_searcher.h"
#include "ivf_pq/ivf_pq_builder.h"
#include "ivf_pq/ivf_pq_merger.h"
#include "ivf_pq/ivf_pq_searcher.h"
#include "ivf_fast_scan/ivf_fast_scan_builder.h"
#include "ivf_fast_scan/ivf_fast_scan_merger.h"
#include "ivf_fast_scan/ivf_fast_scan_searcher.h"
#include "ivf_rpq/ivf_rpq_index.h"
#include "ivf_rpq/ivf_rpq_builder.h"
#include "ivf_rpq/ivf_rpq_merger.h"
#include "ivf_rpq/ivf_rpq_searcher.h"

// for gpu
DEFINE_uint32(global_gpu_force_control, 2,
              "flag for global_gpu_force_control"); // 0: no control, 1: force enable, 2: force disable
DEFINE_uint32(global_gpu_enable_plus, 0, "flag for global_gpu_enable_plus"); // 0: not enable, 1: enable
DEFINE_string(global_gpu_force_index, "", "flag for global_gpu_force_index"); // force reset enable_gpu when global_gpu_force_control = 0

MERCURY_NAMESPACE_BEGIN(core);

Builder::Pointer AlgorithmFactory::CreateBuilder()
{

    Builder::Pointer builder;
    if (GetAlgorithm() == Ivf) {
        builder.reset(new IvfBuilder());
        if (builder->Init(index_params_) != 0) {
            LOG_ERROR("init ivfbuilder failed.");
            return Builder::Pointer();
        }
        return builder;
    }

    if (GetAlgorithm() == IvfPQ) {
        LOG_INFO("factory begin to create ivfpq builder.");
        builder.reset(new IvfPqBuilder());
        if (builder->Init(index_params_) != 0) {
            LOG_ERROR("init ivf pq builder failed.");
            return Builder::Pointer();
        }

        LOG_INFO("factory create ivfpq builder success.");
        return builder;
    }

    if (GetAlgorithm() == GroupIvf) {
        LOG_INFO("factory begin to create groupivf builder.");
        builder.reset(new GroupIvfBuilder());
        if (builder->Init(index_params_) != 0) {
            LOG_ERROR("init group ivf builder failed.");
            return Builder::Pointer();
        }

        LOG_INFO("factory create group ivf builder success.");
        return builder;
    }

    if (GetAlgorithm() == GroupHnsw) {
        LOG_INFO("factory begin to create grouphnsw builder.");
        builder.reset(new GroupHnswBuilder());
        if (builder->Init(index_params_) != 0) {
            LOG_ERROR("init group hnsw builder failed.");
            return Builder::Pointer();
        }

        LOG_INFO("factory create group hnsw builder success.");
        return builder;
    }

    if (GetAlgorithm() == DiskVamana) {
        LOG_INFO("factory begin to create diskvamana builder.");
        builder.reset(new DiskVamanaBuilder());
        if (builder->Init(index_params_) != 0) {
            LOG_ERROR("init disk vamana builder failed.");
            return Builder::Pointer();
        }

        LOG_INFO("factory create disk vamana builder success.");
        return builder;
    }

    if (GetAlgorithm() == RamVamana) {
        LOG_INFO("factory begin to create ramvamana builder.");
        builder.reset(new RamVamanaBuilder());
        if (builder->Init(index_params_) != 0) {
            LOG_ERROR("init ram vamana builder failed.");
            return Builder::Pointer();
        }

        LOG_INFO("factory create ram vamana builder success.");
        return builder;
    }

    if (GetAlgorithm() == IvfFastScan) {
        LOG_INFO("factory begin to create ivffastscan builder.");
        builder.reset(new IvfFastScanBuilder());
        if (builder->Init(index_params_) != 0) {
            LOG_ERROR("init ivf fastscan builder failed.");
            return Builder::Pointer();
        }

        LOG_INFO("factory create ivf fastscan builder success.");
        return builder;
    }
    
    if (GetAlgorithm() == IvfRpq) {
        LOG_INFO("factory begin to create IvfRpq builder.");
        builder.reset(new IvfRpqBuilder());
        if (builder->Init(index_params_) != 0) {
            LOG_ERROR("init ivf rpq builder failed.");
            return Builder::Pointer();
        }

        LOG_INFO("factory create ivf rpq builder success.");
        return builder;
    }

    if (GetAlgorithm() == GroupIvfPq) {
        LOG_INFO("factory begin to create groupivfpq builder.");
        builder.reset(new GroupIvfPqBuilder());
        if (builder->Init(index_params_) != 0) {
            LOG_ERROR("init group ivf pq builder failed.");
            return Builder::Pointer();
        }

        LOG_INFO("factory create group ivf pq builder success.");
        return builder;
    }

    if (GetAlgorithm() == GroupIvfPqSmall) {
        LOG_INFO("factory begin to create groupivfpqsmall builder.");
        builder.reset(new GroupIvfPqSmallBuilder());
        if (builder->Init(index_params_) != 0) {
            LOG_ERROR("init group ivf pq small builder failed.");
            return Builder::Pointer();
        }

        LOG_INFO("factory create group ivf pq small builder success.");
        return builder;
    }

    return builder;
}

int DoCreateIndex(Index *index, IndexParams &index_params, bool for_load)
{
    if (!for_load) {
        return ((IvfIndex *)index)->Create(index_params);
    }

    ((IvfIndex *)index)->SetIndexParams(index_params);
    return 0;
}

Index::Pointer AlgorithmFactory::CreateIndex(bool for_load)
{
    Index::Pointer index;
    if (GetAlgorithm() == Ivf) {
        index.reset(new IvfIndex());
        if (DoCreateIndex(index.get(), index_params_, for_load) != 0) {
            LOG_ERROR("init ivfindex failed.");
            return Index::Pointer();
        }
        return index;
    }

    if (GetAlgorithm() == IvfPQ) {
        index.reset(new IvfPqIndex());
        if (DoCreateIndex(index.get(), index_params_, for_load) != 0) {
            LOG_ERROR("init ivfpqindex failed.");
            return Index::Pointer();
        }
        return index;
    }

    if (GetAlgorithm() == GroupIvf) {
        index.reset(new GroupIvfIndex());
        if (DoCreateIndex(index.get(), index_params_, for_load) != 0) {
            LOG_ERROR("init groupivfindex failed.");
            return Index::Pointer();
        }
        return index;
    }

    if (GetAlgorithm() == GroupHnsw) {
        index.reset(new GroupHnswIndex());
        if (DoCreateIndex(index.get(), index_params_, for_load) != 0) {
            LOG_ERROR("init groupHnswindex failed.");
            return Index::Pointer();
        }
        return index;
    }

    if (GetAlgorithm() == DiskVamana) {
        index.reset(new DiskVamanaIndex());
        if (DoCreateIndex(index.get(), index_params_, for_load) != 0) {
            LOG_ERROR("init DiskVamanaIndex failed.");
            return Index::Pointer();
        }
        return index;
    }

    if (GetAlgorithm() == RamVamana) {
        index.reset(new RamVamanaIndex());
        if (DoCreateIndex(index.get(), index_params_, for_load) != 0) {
            LOG_ERROR("init RamVamanaIndex failed.");
            return Index::Pointer();
        }
        return index;
    }

    if (GetAlgorithm() == IvfFastScan) {
        index.reset(new IvfFastScanIndex());
        if (DoCreateIndex(index.get(), index_params_, for_load) != 0) {
            LOG_ERROR("init IvfFastScanIndex failed.");
            return Index::Pointer();
        }
        return index;
    }

    if (GetAlgorithm() == IvfRpq) {
        index.reset(new IvfRpqIndex());
        if (DoCreateIndex(index.get(), index_params_, for_load) != 0) {
            LOG_ERROR("init IvfRpqIndex failed.");
            return Index::Pointer();
        }
        return index;
    }
    
    if (GetAlgorithm() == GroupIvfPq) {
        index.reset(new GroupIvfPqIndex());
        if (DoCreateIndex(index.get(), index_params_, for_load) != 0) {
            LOG_ERROR("init groupIvfPqindex failed.");
            return Index::Pointer();
        }
        return index;
    }

    if (GetAlgorithm() == GroupIvfPqSmall) {
        index.reset(new GroupIvfPqSmallIndex());
        if (DoCreateIndex(index.get(), index_params_, for_load) != 0) {
            LOG_ERROR("init groupIvfPqSmallindex failed.");
            return Index::Pointer();
        }
        return index;
    }

    return index;
}

Searcher::Pointer AlgorithmFactory::CreateSearcher(bool in_mem)
{
    Searcher::Pointer searcher;
    if (GetAlgorithm() == Ivf) {
        searcher.reset(new IvfSearcher());
        if (searcher->Init(index_params_) != 0) {
            LOG_ERROR("init ivf searcher failed.");
            return Searcher::Pointer();
        }
        return searcher;
    }

    if (GetAlgorithm() == IvfPQ) {
        searcher.reset(new IvfPqSearcher());
        if (searcher->Init(index_params_) != 0) {
            LOG_ERROR("init ivf pq searcher failed.");
            return Searcher::Pointer();
        }
        return searcher;
    }

#ifdef ENABLE_GPU_IN_MERCURY_
    if (((GetAlgorithm() == GroupIvf || GetAlgorithm() == GroupIvfPq 
        || GetAlgorithm() == GroupIvfPqSmall || GetAlgorithm() == IvfRpq) && !in_mem) 
        || (GetAlgorithm() == GroupIvf && in_mem)) {
        bool enable_gpu = false;
        if (!in_mem) {
            enable_gpu = index_params_.has(PARAM_VECTOR_ENABLE_GPU) ? index_params_.getBool(PARAM_VECTOR_ENABLE_GPU) : false;
        }
        else {
            enable_gpu = index_params_.has(PARAM_VECTOR_ENABLE_GPU_INMEM) ? index_params_.getBool(PARAM_VECTOR_ENABLE_GPU_INMEM) : false;
        }
            
        bool enable_gpu_plus = index_params_.has(PARAM_VECTOR_ENABLE_GPU_PLUS)
                                   ? index_params_.getBool(PARAM_VECTOR_ENABLE_GPU_PLUS)
                                   : false;
        // priority: enable_gpu > enable_gpu_plus
        if (FLAGS_global_gpu_enable_plus == 1 && enable_gpu_plus) {
            enable_gpu = enable_gpu_plus;
        }
        
        // first priority
        if (index_params_.has(PARAM_VECTOR_INDEX_NAME)) {
            std::string index_name = index_params_.getString(PARAM_VECTOR_INDEX_NAME)+","; // NB: add ,
            if (FLAGS_global_gpu_force_index.find(index_name) != std::string::npos) {
                enable_gpu = true;
            }
        }

        if (FLAGS_global_gpu_force_control == 1 || (FLAGS_global_gpu_force_control == 0 && enable_gpu)) {
            if (GetAlgorithm() == GroupIvf) {
                searcher.reset(new GpuGroupIvfSearcher());
            }
            else if (GetAlgorithm() == GroupIvfPq){
                searcher.reset(new GpuGroupIvfPqSearcher());
            }
            else if (GetAlgorithm() == GroupIvfPqSmall){
                searcher.reset(new GpuGroupIvfPqSmallSearcher());
            } 
            else if (GetAlgorithm() == IvfRpq){
                searcher.reset(new GpuIvfRpqSearcher());
            }
        } else {
            // FLAGS_global_gpu_force_control == 2
            // FLAGS_global_gpu_force_control == 0 && enable_gpu == false
            if (GetAlgorithm() == GroupIvf) {
                searcher.reset(new GroupIvfSearcher());
            }
            else if (GetAlgorithm() == GroupIvfPq){
                searcher.reset(new GroupIvfPqSearcher());
            }
            else if (GetAlgorithm() == GroupIvfPqSmall){
                searcher.reset(new GroupIvfPqSmallSearcher());
            }
            else if (GetAlgorithm() == IvfRpq){
                searcher.reset(new IvfRpqSearcher());
            }
        }
        if (searcher->Init(index_params_) != 0) {
            LOG_ERROR("init group ivf searcher failed, %d, %d", FLAGS_global_gpu_force_control, enable_gpu);
            return Searcher::Pointer();
        }
        return searcher;
    }
#endif

    if (GetAlgorithm() == GroupIvf) {
        searcher.reset(new GroupIvfSearcher());
        if (searcher->Init(index_params_) != 0) {
            LOG_ERROR("init group ivf searcher failed.");
            return Searcher::Pointer();
        }
        return searcher;
    }

    if (GetAlgorithm() == GroupHnsw) {
        searcher.reset(new GroupHnswSearcher());
        if (searcher->Init(index_params_) != 0) {
            LOG_ERROR("init group ivf searcher failed.");
            return Searcher::Pointer();
        }
        return searcher;
    }

    if (GetAlgorithm() == DiskVamana) {
        searcher.reset(new DiskVamanaSearcher());
        if (searcher->Init(index_params_) != 0) {
            LOG_ERROR("init disk vamana searcher failed.");
            return Searcher::Pointer();
        }
        return searcher;
    }

    if (GetAlgorithm() == RamVamana) {
        searcher.reset(new RamVamanaSearcher());
        if (searcher->Init(index_params_) != 0) {
            LOG_ERROR("init ram vamana searcher failed.");
            return Searcher::Pointer();
        }
        return searcher;
    }

    if (GetAlgorithm() == IvfFastScan) {
        searcher.reset(new IvfFastScanSearcher());
        if (searcher->Init(index_params_) != 0) {
            LOG_ERROR("init ivf fastscan searcher failed.");
            return Searcher::Pointer();
        }
        return searcher;
    }

    if (GetAlgorithm() == IvfRpq) {
        searcher.reset(new IvfRpqSearcher());
        if (searcher->Init(index_params_) != 0) {
            LOG_ERROR("init ivf rpq searcher failed.");
            return Searcher::Pointer();
        }
        return searcher;
    }

    if (GetAlgorithm() == GroupIvfPq) {
        searcher.reset(new GroupIvfPqSearcher());
        if (searcher->Init(index_params_) != 0) {
            LOG_ERROR("init group ivf pq searcher failed.");
            return Searcher::Pointer();
        }
        return searcher;
    }

    if (GetAlgorithm() == GroupIvfPqSmall) {
        searcher.reset(new GroupIvfPqSmallSearcher());
        if (searcher->Init(index_params_) != 0) {
            LOG_ERROR("init group ivf pq small searcher failed.");
            return Searcher::Pointer();
        }
        return searcher;
    }

    return Searcher::Pointer();
}

Merger::Pointer AlgorithmFactory::CreateMerger()
{
    Merger::Pointer merger;
    if (GetAlgorithm() == Ivf) {
        merger.reset(new IvfMerger());
        if (merger->Init(index_params_) != 0) {
            LOG_ERROR("init ivf merger failed.");
            return Merger::Pointer();
        }
        return merger;
    }

    if (GetAlgorithm() == IvfPQ) {
        merger.reset(new IvfPqMerger());
        if (merger->Init(index_params_) != 0) {
            LOG_ERROR("init ivf pq merger failed.");
            return Merger::Pointer();
        }
        return merger;
    }

    if (GetAlgorithm() == GroupIvf) {
        merger.reset(new GroupIvfMerger());
        if (merger->Init(index_params_) != 0) {
            LOG_ERROR("init group ivf merger failed.");
            return Merger::Pointer();
        }
        return merger;
    }

    if (GetAlgorithm() == GroupHnsw) {
        merger.reset(new GroupHnswMerger());
        if (merger->Init(index_params_) != 0) {
            LOG_ERROR("init group ivf searcher failed.");
            return Merger::Pointer();
        }
        return merger;
    }

    if (GetAlgorithm() == IvfFastScan) {
        merger.reset(new IvfFastScanMerger());
        if (merger->Init(index_params_) != 0) {
            LOG_ERROR("init group ivf fastscan merger failed.");
            return Merger::Pointer();
        }
        return merger;
    }

    if (GetAlgorithm() == IvfRpq) {
        merger.reset(new IvfRpqMerger());
        if (merger->Init(index_params_) != 0) {
            LOG_ERROR("init group ivf rpq merger failed.");
            return Merger::Pointer();
        }
        return merger;
    }

    if (GetAlgorithm() == GroupIvfPq) {
        merger.reset(new GroupIvfPqMerger());
        if (merger->Init(index_params_) != 0) {
            LOG_ERROR("init group ivf pq searcher failed.");
            return Merger::Pointer();
        }
        return merger;
    }

    if (GetAlgorithm() == GroupIvfPqSmall) {
        merger.reset(new GroupIvfPqSmallMerger());
        if (merger->Init(index_params_) != 0) {
            LOG_ERROR("init group ivf pq small searcher failed.");
            return Merger::Pointer();
        }
        return merger;
    }

    return Merger::Pointer();
}

MERCURY_NAMESPACE_END(core);
