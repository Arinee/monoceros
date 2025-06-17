#ifndef __MERCURY_CORE_INDEX_H__
#define __MERCURY_CORE_INDEX_H__

#include <cstdio>
#include <string>
#include "src/core/framework/index_meta.h"
#include "src/core/framework/index_params.h"
#include "src/core/framework/index_package.h"
#include "src/core/framework/index_logger.h"
#include "src/core/framework/index_distance.h"
#include "src/core/framework/index_framework.h"
#include "src/core/framework/utility/bitset_helper.h"
//#include "src/core/common/common_define.h"
#include "src/core/common/common.h"
#include "src/core/utils/array_profile.h"
#include "src/core/utils/hash_table.h"
#include "src/core/algorithm/dump_helper.h"
#include "src/core/framework/deletion_map_retriever.h"

MERCURY_NAMESPACE_BEGIN(core);

inline void decode_vector(const uint8_t* code, float* x, float vmin, float vmax, int d) {
    float vdiff = vmax - vmin;
    for (int i = 0; i < d; ++i) {
        float xi = (code[i] + 0.5f) / 255.0f;
        x[i] = vmin + xi * vdiff;
    }
}

inline void encode_vector(const float* x, uint8_t* code, float vmin, float vmax, int d) {
    float vdiff = vmax - vmin;
    for (int i = 0; i < d; ++i) {
        float xi = (x[i] - vmin) / vdiff;
        if (xi < 0) xi = 0;
        if (xi > 1.0) xi = 1.0;
        code[i] = static_cast<uint8_t>(255 * xi);
    }
}

class Searcher;
class Index
{
public:
    typedef std::shared_ptr<Index> Pointer;
    Index () {};
    /// load index from index package file and index will save this handle
    virtual int Load(const void*, size_t);
    /// add a new vector
    virtual int Add(docid_t doc_id, pk_t pk,
                    const std::string& query_str, 
                    const std::string& primary_key = "");
    /// add a new vector
    int Add(docid_t doc_id, pk_t pk, const void *val, size_t len);
    /// calc doc count with memQuata
    virtual size_t MemQuota2DocCount(size_t memQuota, size_t elemSize) const = 0;
    /// Display the actual class name and some more info
    virtual void Display() const{};
    /// whither index segment full
    virtual bool IsFull() const = 0;

    virtual int64_t UsedMemoryInCurrent() const = 0;

    virtual int Dump(const void*& data, size_t& size) {
        if (DumpHelper::DumpCommon(this, index_package_) != 0) {
            LOG_ERROR("dump into package failed.");
            return -1;
        }

        if (!index_package_.dump(data, size)) {
            LOG_ERROR("Failed to dump package.");
            return -1;
        }

        return 0;
    }

    virtual int DumpCentroid(const void *&data, size_t &size)
    {
        data = nullptr;
        size = 0;
        return 0;
    }

    void SetIndexMeta(const IndexMeta& index_meta) {
        index_meta_ = index_meta;
    }
    const IndexMeta& GetIndexMeta(void) const {
        return index_meta_;
    }

    void SetIndexParams(const IndexParams& index_params) {
        index_params_ = index_params;
        contain_feature_ = index_params_.getBool(PARAM_GENERAL_CONTAIN_FEATURE_PROFILE);
        multi_age_mode_ = index_params_.getBool(PARAM_MULTI_AGE_MODE);
        enable_mips_ = index_params_.getBool(PARAM_ENABLE_MIPS);
        force_half_ = index_params_.getBool(PARAM_ENABLE_FORCE_HALF);
        enable_fine_cluster_ = index_params.getBool(PARAM_ENABLE_FINE_CLUSTER);
        enable_residual_ = index_params.getBool(PARAM_ENABLE_RESIDUAL);
        if (contain_feature_) {
            enable_quantize_ = index_params.getBool(PARAM_ENABLE_QUANTIZE);
        }
    }

    bool ContainFeature() const {
        return contain_feature_;
    }
    bool MultiAgeMode() const {
        return multi_age_mode_;
    }
    bool EnableMips() const {
        return enable_mips_;
    }
    bool ForceHalf() const {
        return force_half_;
    }
    bool EnableFineCluster() const {
        return enable_fine_cluster_;
    }
    bool EnableResidual() const {
        return enable_residual_;
    }
    bool EnableQuantize() const {
        return enable_quantize_;
    }
    float GetVmin() const {
        return vmin_;
    }
    float GetVmax() const {
        return vmax_;
    }
    void SetVmin(float val) {
        vmin_ = val;
    }
    void SetVmax(float val) {
        vmax_ = val;
    }

    const IndexParams& GetIndexParams() const {
        return index_params_;
    }

    uint64_t GetPk(const docid_t& docid) const {
        return *((uint64_t *)pk_profile_.getInfo(docid));
    }

    const ArrayProfile& GetPkProfile() const {
        return pk_profile_;
    }

    HashTable<pk_t, docid_t>& GetIdMap() {
        return id_map_;
    }

    virtual uint64_t GetMaxDocNum() const  = 0;

    virtual float GetRankScore(docid_t doc_id) = 0;

    /// Read doc num
    virtual size_t GetDocNum(void) const  = 0;

    virtual int CopyInit(const Index*, size_t);

    bool WithPk() const {
        return index_params_.getBool(PARAM_WITH_PK);
    }

    std::string& GetMetaData() {
        return meta_data_;
    }

    virtual void SetBaseDocid(exdocid_t base_docid)
    {
        base_docid_ = base_docid;
    }

    virtual void SetDeletionMapRetriever(const DeletionMapRetriever& retriever) {
        deletion_map_retriever_ = retriever;
    }

    virtual void SetIsGpu(bool is_gpu) {
        is_gpu_ = is_gpu;
    }

    virtual void SetSearcher(Searcher* searcher) {
        searcher_ = searcher;
    }

    virtual void SetInMem(bool in_mem) {
        in_mem_ = in_mem;
    }

protected:
    int Create(size_t max_doc_num);

protected:
    IndexPackage index_package_;
    /// index meta, must alloc new when create a index
    IndexMeta index_meta_;
    /// index params
    IndexParams index_params_;
    // docid -> pk
    ArrayProfile pk_profile_;
    std::vector<char> pk_profile_base_;
    // pk -> docid
    HashTable<pk_t, docid_t> id_map_;
    std::vector<char> id_map_base_;
    uint64_t max_doc_num_;

    std::string meta_data_;

    exdocid_t base_docid_;

    bool is_gpu_ = false;
    bool in_mem_ = false;

    Searcher* searcher_ = nullptr;

protected:
    DeletionMapRetriever deletion_map_retriever_;

private:
    bool contain_feature_ = false;
    bool multi_age_mode_ = false;
    bool enable_mips_ = false;
    bool force_half_ = false;
    bool enable_fine_cluster_ = false;
    bool enable_residual_ = false;
    bool enable_quantize_ = false;
    float vmin_ = HUGE_VAL;
    float vmax_ = -HUGE_VAL;
};

MERCURY_NAMESPACE_END(core);

#endif // __MERCURY_CORE_INDEX_H__
