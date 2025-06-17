#include "dump_helper.h"
#include "ivf/ivf_index.h"
#include "ivf_pq/ivf_pq_index.h"

MERCURY_NAMESPACE_BEGIN(core);

int DumpHelper::DumpCommon(Index* index, IndexPackage& index_package) {
    if (index == nullptr) {
        return -1;
    }

    std::string& meta_data = index->GetMetaData();
    index->GetIndexMeta().serialize(&meta_data);
    index_package.emplace(COMPONENT_FEATURE_META, meta_data.data(), meta_data.size());
    if (index->WithPk()) {
        index_package.emplace(COMPONENT_PK_PROFILE, index->GetPkProfile().getHeader(),
                              index->GetPkProfile().getHeader()->capacity);
        index_package.emplace(COMPONENT_IDMAP, index->GetIdMap().getBase(), index->GetIdMap().size());
    }

    return 0;
}

int DumpHelper::DumpIvf(IvfIndex* ivf_index, IndexPackage& index_package) {
    if (DumpCommon(ivf_index, index_package) != 0) {
        LOG_ERROR("dump common failed.");
        return -1;
    }

    std::string& rough_matrix = ivf_index->GetRoughMatrix();
    ivf_index->GetCentroidResource().dumpRoughMatrix(rough_matrix);
    index_package.emplace(COMPONENT_ROUGH_MATRIX, rough_matrix.data(), rough_matrix.size());

    index_package.emplace(COMPONENT_COARSE_INDEX, ivf_index->GetCoarseIndex().GetBasePtr(),
                          ivf_index->GetCoarseIndex().getHeader()->capacity);
    index_package.emplace(COMPONENT_SLOT_INDEX_PROFILE, ivf_index->GetSlotIndexProfile().getHeader(),
                          ivf_index->GetSlotIndexProfile().getHeader()->capacity);
    return 0;
}

int DumpHelper::DumpIvfPq(IvfPqIndex* index, IndexPackage& index_package) {
    if (DumpIvf(index, index_package) != 0) {
        LOG_ERROR("dump ivf failed.");
        return -1;
    }

    std::string& integrate_matrix_str = index->GetIntegrateMatrixStr();
    index->GetCentroidResource().dumpIntegrateMatrix(integrate_matrix_str);
    index_package.emplace(COMPONENT_INTEGRATE_MATRIX, integrate_matrix_str.data(),
                          integrate_matrix_str.size());

    index_package.emplace(COMPONENT_PQ_CODE_PROFILE, index->GetPqCodeProfile().getHeader(),
                          index->GetPqCodeProfile().getHeader()->capacity);
    return 0;
}

MERCURY_NAMESPACE_END(core);
