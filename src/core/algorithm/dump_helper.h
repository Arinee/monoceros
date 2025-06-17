#ifndef __MERCURY_CORE_DUMP_UTIL_H__
#define __MERCURY_CORE_DUMP_UTIL_H__

#include "src/core/common/common.h"
#include "src/core/framework/index_package.h"

MERCURY_NAMESPACE_BEGIN(core);

class Index;
class IvfIndex;
class IvfPqIndex;

class DumpHelper {
public:
    int static DumpCommon(Index* index, IndexPackage& index_package);

    int static DumpIvf(IvfIndex* ivf_index, IndexPackage& index_package);

    int static DumpIvfPq(IvfPqIndex* index, IndexPackage& index_package);
};

MERCURY_NAMESPACE_END(core);

#endif
