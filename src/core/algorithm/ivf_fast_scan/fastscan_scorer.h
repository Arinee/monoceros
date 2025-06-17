/// Copyright (c) 2024, xiaohongshu Inc. All rights reserved.
/// Author: haiming <shiyang1@xiaohongshu.com>
/// Created: 2024-08-28 18:11

#pragma once

#include "look_up_table.h"
#include "src/core/utils/array_profile.h"
#include "fastscan_kernel.h"
#include "fastscan_index.h"
#include <memory>
#include <string>
#include <vector>

MERCURY_NAMESPACE_BEGIN(core);

using namespace mercury::internal;

#ifdef SWIG
#define ALIGNED(x)
#else
#define ALIGNED(x) __attribute__((aligned(x)))
#endif

class FastScanScorer 
{
public:
    typedef std::shared_ptr<FastScanScorer> Pointer;
public:
    FastScanScorer() {}

    bool score(const uint8_t *lut, const uint8_t *packedCodes,
                size_t nsq, float_t scale, float_t bias,
                size_t start_index, size_t doc_count,
                std::vector<DistNode>* dist_nodes) {
        const uint8_t* codes = packedCodes;
        const uint8_t* qlut = lut;
        size_t code32_offset = nsq * 16UL; // code32_offset = 32 * fragment_num / 2
        size_t roundUp32DocCount = FastScanIndex::ROUNDUP32(doc_count);
        size_t cur_index = start_index;
        for (size_t i = 0; i < roundUp32DocCount; i += 32) {
            size_t left_count = start_index + doc_count - cur_index;
            size_t dist_count = left_count < 32 ? left_count : 32;
            accumulate(codes, qlut, nsq, scale, bias, cur_index, dist_count, dist_nodes);
            cur_index += 32;
            codes += code32_offset;
        }
        return true;
    }

    static bool accumulate(const uint8_t* codes32, const uint8_t* qlut,
                            size_t nsq, float_t scale, float_t bias, 
                            size_t cur_index, size_t dist_count,
                            std::vector<DistNode>* dist_nodes) {
        simd16uint16 accu[4];
        // Initialize accumulators
        for (int b = 0; b < 4; b++) {
            accu[b].clear();
        }
        // Main loop over sub-quantizers
        for (size_t sq = 0; sq < nsq; sq += 2) {
            // Load codes and LUTs
            simd32uint8 c(codes32);
            codes32 += 32;

            simd32uint8 mask(0xf);
            simd32uint8 chi = simd32uint8(simd16uint16(c) >> 4) & mask;
            simd32uint8 clo = c & mask;

            // Load LUTs for 2 quantizers
            simd32uint8 lut(qlut);
            qlut += 32;

            // Perform lookups
            simd32uint8 res0 = lut.lookup_2_lanes(clo);
            simd32uint8 res1 = lut.lookup_2_lanes(chi);

            // Accumulate distances
            accu[0] += simd16uint16(res0);
            accu[1] += simd16uint16(res0) >> 8;

            accu[2] += simd16uint16(res1);
            accu[3] += simd16uint16(res1) >> 8;
        }
        // Final distance computation
        accu[0] -= accu[1] << 8;
        simd16uint16 dis0 = combine2x2(accu[0], accu[1]);

        accu[2] -= accu[3] << 8;
        simd16uint16 dis1 = combine2x2(accu[2], accu[3]);

        ALIGNED(32) uint16_t d32[32];
        dis0.store(d32);
        dis1.store(d32 + 16);

        for (size_t i = cur_index; i < cur_index + dist_count; i++) {
            dist_nodes->at(i).dist = d32[i - cur_index] * (1 / scale) + bias;
        }
        return true;
    }

};

MERCURY_NAMESPACE_END(core);
