#ifndef __MERCURY_CORE_MOCKED_VECTOR_READER_H__
#define __MERCURY_CORE_MOCKED_VECTOR_READER_H__

#include "group_ivf_index.h"

MERCURY_NAMESPACE_BEGIN(core);

class MockedVectorReader
{
public:
    MockedVectorReader(const GroupIvfIndex::Pointer index) : index_(index) {
        memset(mock_buf, 0, 1024);
    }

    bool ReadProfile(docid_t docid, const void*& base) const {
        base = index_->GetFeatureProfile().getInfo(docid);
        return true;
    }

    bool ReadConstant(docid_t docid, const void*& base) const {
        base = (void *)mock_buf;
        return true;
    }
private:
    const GroupIvfIndex::Pointer index_;
    char mock_buf[1024];
};

MERCURY_NAMESPACE_END(core);

#endif // __MERCURY_CORE_MOCKED_VECTOR_READER_H__
