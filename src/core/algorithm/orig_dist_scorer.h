/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     orig_dist_scorer.h
 *   \author   qiuming
 *   \date     Jan 2019
 *   \version  1.0.0
 *   \brief    interface of OrigDistScorer
 */

#ifndef __MERCURY_ORIG_DIST_SCORER_H__
#define __MERCURY_ORIG_DIST_SCORER_H__

#include "src/core/framework/index_framework.h"
#include "general_search_context.h"
#include <memory>
#include <string>
#include <vector>

MERCURY_NAMESPACE_BEGIN(core);

class OrigDistScorer
{
public:
    typedef std::shared_ptr<OrigDistScorer> Pointer;

    class Factory 
    {
    public:
        Factory()
        {}
        int Init(IndexMeta index_meta)
        {
            _indexMeta = index_meta;
            return 0;
        }

        OrigDistScorer Create(GeneralSearchContext* context) const
        {
            OrigDistScorer s;
            s._elemSize = _indexMeta.sizeofElement();
            s._measure = _indexMeta.measure();
            s._dimension = _indexMeta.dimension();
            s._type = _indexMeta.type();
            if (context->getSearchMethod() != IndexDistance::kMethodUnknown) {
                s._measure = IndexDistance::EmbodyMeasure(context->getSearchMethod());
            }
            return s;
        }

        OrigDistScorer Create() const
            {
                OrigDistScorer s;
                s._elemSize = _indexMeta.sizeofElement();
                s._measure = _indexMeta.measure();
                s._dimension = _indexMeta.dimension();
                s._type = _indexMeta.type();
                return s;
            }
    private:
        IndexMeta _indexMeta;
    };

public:
    OrigDistScorer()
        :_elemSize(0)
        {}
    score_t Score(const void* goods, const void* query) {
        std::unique_ptr<char[]> matrix_pointer;
        if (_type == IndexMeta::FeatureTypes::kTypeBinary) {
            char *vec_matrix = new char[_elemSize];
            matrix_pointer.reset(vec_matrix);
            std::memset(vec_matrix, 0, _elemSize);
            const float *vec = reinterpret_cast<const float*>(goods);
            for (size_t i = 0; i < _dimension; ++i) {
                float value = *(vec + i);
                uint32_t pos = i % 8;
                if (std::abs(value - 0.0) < 1e-5) {
                    continue;
                } else if (std::abs(value - 1.0) < 1e-5) {
                    char mask = 1u << (7u - pos);
                    vec_matrix[i / 8] |= mask;
                } else {
                    LOG_ERROR("invalid vector value: %f", value);
                    return 0.0;
                }
            }
            goods = reinterpret_cast<const void*>(vec_matrix);
        }
        return _measure(query, goods, _elemSize);
    }

    score_t Score(const std::vector<const void*>& doc_features, const std::vector<const void*>& queries) {
        if (doc_features.size() == 0 || queries.size() == 0) {
            return 0;
        }

        if (doc_features.size() == 1) {
            return Min(doc_features.at(0), queries);
        }

        if (queries.size() == 1) {
            return Min(queries.at(0), doc_features);
        }

        float score = 0.0;
        for (size_t i = 0; i < queries.size(); i++) {
            score += Min(queries.at(i), doc_features);
        }
        return score;
    }

    score_t Score(const void* left, const std::vector<const void*>& right) {
        return Min(left, right);
    }

    score_t Min(const void* left, const std::vector<const void*>& right) {
        float min = std::numeric_limits<float>::max();
        for (size_t i = 0; i < right.size(); i++) {
            float score = Score(left, right.at(i));
            if (score < min) {
                min = score;
            }
        }
        return min;
    }
private:
    size_t _elemSize;
    size_t _dimension;
    IndexDistance::Measure _measure;
    IndexMeta::FeatureTypes _type;
};

MERCURY_NAMESPACE_END(core);
#endif // __MERCURY_ORIG_DIST_SCORER_H__
