/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     reservoir_sample.h
 *   \author   yunheng@xiaohongshu.com
 *   \date     Jan 2019
 *   \version  1.0.0
 *   \brief    Interface of Reservoir Sampling
 */

#ifndef __MERCURY_CLUSTER_RESERVOIR_SAMPLE_H__
#define __MERCURY_CLUSTER_RESERVOIR_SAMPLE_H__

#include "framework/utility/reservoir_sampler.h"
#include "cluster_feature.h"
#include <random>

namespace mercury {

//! Reservoir sampling
static inline void ReservoirSample(const ClusterFeatureAny *feats, size_t count,
                                   size_t sample_count,
                                   std::vector<ClusterFeatureAny> *out)
{
    mercury::ReservoirSampler<ClusterFeatureAny> sampler(sample_count);

    // Reservoir Sampling
    for (size_t i = 0; i < count; ++i) {
        sampler.fill(feats[i]);
    }
    *out = std::move(sampler.pool());
}

//! Reservoir sampling
static inline void ReservoirSample(const std::vector<ClusterFeatureAny> &feats,
                                   size_t sample_count,
                                   std::vector<ClusterFeatureAny> *out)
{
    return ReservoirSample(feats.data(), feats.size(), sample_count, out);
}

} // namespace mercury

#endif // __MERCURY_CLUSTER_RESERVOIR_SAMPLE_H__ 
