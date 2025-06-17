/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     ivf_seeker.h
 *   \author   qiuming
 *   \date     Jan 2019
 *   \version  1.0.0
 *   \brief    interface of ivf seeker
 */

#ifndef __MERCURY_IVF_SEEKER_H__
#define __MERCURY_IVF_SEEKER_H__

#include "framework/index_framework.h"
#include "posting_iterator.h"
#include "coarse_index.h"
#include "index.h"
#include <memory>
#include <string>
#include <vector>

namespace mercury 
{

class CentroidQuantizer;
class IndexMeta;
class CoarseIndex;
class GeneralSearchContext;

/*
 * seek by ivf index
 */
class IvfSeeker 
{
public:
    typedef std::shared_ptr<IvfSeeker> Pointer;
public:
    explicit IvfSeeker(std::vector<float> coarseScanRatios)
        : centroid_quantizer_(nullptr),
        coarse_index_(nullptr),
        _coarseScanRatios(coarseScanRatios)
    {
        if (_coarseScanRatios.size() != 2) {
            LOG_ERROR("coarseScanRatios must contain levelScanRatio and nprobeRatio");
        }
    }
    int Init(Index *index);
    std::vector<CoarseIndex::PostingIterator> 
        Seek(const void *query, size_t bytes, GeneralSearchContext *context);

private:
    CentroidQuantizer *centroid_quantizer_;
    CoarseIndex *coarse_index_;
    IndexMeta index_meta_;
    std::vector<float> _coarseScanRatios;
};

} // namespace mercury

#endif // __MERCURY_IVF_SEEKER_H__
