#ifndef __PQ_INDEX_ERROR_H__
#define __PQ_INDEX_ERROR_H__

#include "framework/index_error.h"

namespace mercury
{

INDEX_ERROR_CODE_DECLARE(PQCookbook); // PQCookbook
INDEX_ERROR_CODE_DECLARE(RemoveSegment); // RemoveSegment
INDEX_ERROR_CODE_DECLARE(FlushSegment); // FlushSegment
INDEX_ERROR_CODE_DECLARE(BuildIndex); // BuildIndex
INDEX_ERROR_CODE_DECLARE(CentroidResource); // CentroidResource
INDEX_ERROR_CODE_DECLARE(Holder); // Holder
INDEX_ERROR_CODE_DECLARE(CreateIndex); // CreateIndex

}; // namepace mercury

#endif //__PQ_INDEX_ERROR_H__
