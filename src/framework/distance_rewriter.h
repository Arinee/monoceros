#ifndef __MERCURY_DISTANCE_REWRITER_H__
#define __MERCURY_DISTANCE_REWRITER_H__

#include "index_meta.h"
#include "search_result.h"

namespace mercury {

class DistanceRewriter {
public:
    DistanceRewriter(const mercury::IndexMeta &meta) 
        : _normalizer(meta.method())
    {
    }

    DistanceRewriter(mercury::IndexDistance::Methods method) 
        : _normalizer(method)
    {
    }

    void operator()(mercury::SearchResult &doc) const
    {
        doc.score = _normalizer(doc.score);
    }

private:
    mercury::IndexDistance::Normalizer _normalizer;
};

}; // namespace mercury

#endif //__MERCURY_DISTANCE_REWRITER_H__
