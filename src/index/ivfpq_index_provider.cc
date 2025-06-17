#include "ivfpq_index_provider.h"

using namespace std;
using namespace mercury;

const uint16_t * IvfpqIndexProvider::getProduct(gloid_t gloid)
{
    segid_t segid = GET_SEGID(gloid);
    docid_t docid = GET_DOCID(gloid);
    if (likely(segid < _segments.size())) {
        auto* pq_index= (IndexIvfpq*)_segments[segid].get();
        return reinterpret_cast<const uint16_t *>(pq_index->_pqcodeProfile->getInfo(docid));
    } else {
        return nullptr;
    }
}
