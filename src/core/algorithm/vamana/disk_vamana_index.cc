#include "disk_vamana_index.h"
#include "src/core/utils/index_meta_helper.h"
#include "src/core/common/common.h"

#define READ_INT(stream, val) stream.read((char *)&val, sizeof(int))
#define READ_U64(stream, val) stream.read((char *)&val, sizeof(uint64_t))
#define READ_U32(stream, val) stream.read((char *)&val, sizeof(uint32_t))
#define READ_UNSIGNED(stream, val) stream.read((char *)&val, sizeof(unsigned))

// define a global mutex to for io_context to prevent race
static std::mutex g_io_mutex;

MERCURY_NAMESPACE_BEGIN(core);

DiskVamanaIndex::DiskVamanaIndex()
    : R_(64),
      L_(100),
      alpha_(defaults::ALPHA),
      is_saturated_(false),
      max_occlusion_(defaults::MAX_OCCLUSION_SIZE),
      T_(1),
      max_shard_data_num_(INT_MAX),
      _pq_table_populated(false),
      _use_half(false),
      reader(nullptr),
      _k_base(0)
{}

DiskVamanaIndex::~DiskVamanaIndex()
{
    std::lock_guard<std::mutex> lock(g_io_mutex);
    LOG_INFO("Destructing disk vamana index ...");
    if (reader != nullptr) {
        LOG_INFO("Clearing scratch ...");
        ScratchStoreManager<SSDThreadData> manager(this->_thread_data);
        manager.destroy();
        reader->deregister_all_threads();
        reader->close();
        reader = nullptr;
    }
    LOG_INFO("Releasing cached memory ...");
    if (_use_half) {
        if (_centroid_half_data != nullptr) {
            aligned_free(_centroid_half_data);
        }
        _coord_cache_buf_half.clear();
        _coord_cache_buf_half.shrink_to_fit();
    } else {
        if (_centroid_data != nullptr) {
            aligned_free(_centroid_data);
        }
        _coord_cache_buf.clear();
        _coord_cache_buf.shrink_to_fit();
    }
    if (_nhood_cache_buf) {
        delete[] _nhood_cache_buf;
    }
    LOG_INFO("DiskVamanaIndex Destructed");
}

int DiskVamanaIndex::Create(IndexParams& index_params) {
    Index::SetIndexParams(index_params);

    if (!IndexMetaHelper::parseFrom(index_params.getString(PARAM_DATA_TYPE),
                                    index_params.getString(PARAM_METHOD),
                                    index_params.getUint64(PARAM_DIMENSION),
                                    index_meta_)) {
        LOG_ERROR("Failed to init DiskVamana index meta.");
        return -1;
    }

    if (!IndexMetaHelper::parseFrom("float",
                                    "L2",
                                    index_params.getUint64(PARAM_DIMENSION),
                                    index_meta_L2_)) {
        LOG_ERROR("Failed to init DiskVamana index meta L2.");
        return -1;
    }

    // determine data type
    if (index_meta_.type() == mercury::core::IndexMeta::kTypeHalfFloat) {
        _use_half = true;
    }

    // set Vamana graph build index max degree (R)
    if (index_params.has(PARAM_VAMANA_INDEX_MAX_GRAPH_DEGREE)) {
        R_ = index_params.getUint32(PARAM_VAMANA_INDEX_MAX_GRAPH_DEGREE);
        if (R_ <= 0) {
            LOG_ERROR("mercury.vamana.index.max_graph_degree must larger than 0");
            return -1;
        }
    }

    // set Vamana graph index build max size of search list (L)
    if (index_params.has(PARAM_VAMANA_INDEX_BUILD_MAX_SEARCH_LIST)) {
        L_ = index_params.getUint32(PARAM_VAMANA_INDEX_BUILD_MAX_SEARCH_LIST);
        if (L_ <= 0) {
            LOG_ERROR("mercury.vamana.index.build.max_search_list must larger than 0");
            return -1;
        }
    }

    // set Vamana graph index build alpha
    if (index_params.has(PARAM_VAMANA_INDEX_BUILD_ALPHA)) {
        alpha_ = index_params.getFloat(PARAM_VAMANA_INDEX_BUILD_ALPHA);
        if (alpha_ <= 1.0) {
            LOG_ERROR("mercury.vamana.index.build.alpha must larger than 1.0");
            return -1;
        }
    }

    // set Vamana graph index build is saturated
    if (index_params.has(PARAM_VAMANA_INDEX_BUILD_IS_SATURATED)) {
        is_saturated_ = index_params.getBool(PARAM_VAMANA_INDEX_BUILD_IS_SATURATED);
    }

    // set Vamana graph index build max occlusion size
    if (index_params.has(PARAM_VAMANA_INDEX_BUILD_MAX_OCCLUSION)) {
        max_occlusion_ = index_params.getUint32(PARAM_VAMANA_INDEX_BUILD_MAX_OCCLUSION);
        if (max_occlusion_ <= 0) {
            LOG_ERROR("mercury.vamana.index.build.max_occlusion must larger than 0");
            return -1;
        }
    }

    // set Vamana graph index build max shard data num
    if (index_params.has(PARAM_VAMANA_INDEX_BUILD_MAX_SHARD_DATA_NUM)) {
        max_shard_data_num_ = index_params.getUint32(PARAM_VAMANA_INDEX_BUILD_MAX_SHARD_DATA_NUM);
        if (max_shard_data_num_ <= 0) {
            LOG_ERROR("mercury.vamana.index.build.max_shard_data_num must larger than 0");
            return -1;
        }
    }

    // set Vamana graph index build duplicate factor (default is 2)
    if (index_params.has(PARAM_VAMANA_INDEX_BUILD_DUPLICATE_FACTOR)) {
        _k_base = index_params.getUint32(PARAM_VAMANA_INDEX_BUILD_DUPLICATE_FACTOR);
        if (_k_base <= 1) {
            LOG_ERROR("mercury.vamana.index.build.duplicate_factor must larger than 1");
            return -1;
        }
    } else {
        _k_base = defaults::K_BASE;
    }

    // set Vamana graph index build thread num
    if (index_params.has(PARAM_VAMANA_INDEX_BUILD_THREAD_NUM)) {
        T_ = index_params.getUint32(PARAM_VAMANA_INDEX_BUILD_THREAD_NUM);
        if (T_ <= 0) {
            LOG_ERROR("mercury.vamana.index.build.thread_num must larger than 0");
            return -1;
        }
    }

    //设置doc数量
    size_t max_build_num = index_params.getUint64(PARAM_GENERAL_MAX_BUILD_NUM);
    if (max_build_num > 0) {
        max_doc_num_ = max_build_num;
    } else {
        LOG_ERROR("Not set param: mercury.general.index.max_build_num");
        return -1;
    }

    data_dim_ = index_meta_.dimension();

    aligned_dim_ = ROUND_UP(data_dim_, 8);

    data_size_ = (uint16_t)index_meta_.sizeofElement() / data_dim_;

    size_t max_reserve_degree = (size_t)(defaults::GRAPH_SLACK_FACTOR * 1.05 * R_);
    

    if (!index_params.getBool(PARAM_VAMANA_INDEX_BUILD_IS_PARTITION)) {
        
        auto index_build_params = IndexWriteParametersBuilder(L_, R_)
                                    .with_alpha(alpha_)
                                    .with_saturate_graph(true)
                                    .with_num_threads(T_)
                                    .build();

        auto config = IndexConfigBuilder()
                        .with_dimension(data_dim_)
                        .with_max_points(max_doc_num_)
                        .with_max_reserve_degree(max_reserve_degree)
                        .is_dynamic_index(false)
                        .with_index_write_params(index_build_params)
                        .is_enable_tags(false)
                        .build();

        coarseVamanaIndex_ = std::make_unique<CoarseVamanaIndex>(data_size_, config, index_meta_.method(), _use_half);
        _method = index_meta_.method();
        _measure = IndexDistance::EmbodyMeasure(_method);
    }

    if (index_params.getString(PARAM_TRAIN_DATA_PATH) != "") {
        
        if (!InitPqCentroidMatrix(index_params)) {
            LOG_ERROR("Failed to init pq centroid matrix");
            return -1;
        }

        if (!InitVamanaCentroidMatrix(index_params.getString(PARAM_TRAIN_DATA_PATH))) {
            LOG_ERROR("Failed to init vamana centroid matrix");
            return -1;
        }

        _partition_prefix = std::string(get_current_dir_name()) + "/vamana";

        if (_use_half) {
            _partition_prefix += "_half";
        }

        if (index_params.getBool(PARAM_VAMANA_INDEX_BUILD_IS_PARTITION)) {
            _center_num = GetVamanaCentroidResource().getRoughMeta().centroidNums[0];
            _shard_counts = std::make_unique<size_t[]>(_center_num);
            uint32_t dummy_size = 0;
            uint32_t const_one = 1;
            uint32_t basedim32 = (uint32_t)data_dim_;
            _partition_prefix = std::string(get_current_dir_name()) + "/vamana_partition";
            if (_use_half) {
                _partition_prefix += "_half";
            }
            std::string ori_data_filename = _partition_prefix + "_ori.data";
            _ori_data_num = 0;
            _cached_ori_data_writer = new cached_ofstream(ori_data_filename, BUFFER_SIZE_MID_FOR_CACHED_IO);
            _cached_ori_data_writer->write((char *)&_ori_data_num, sizeof(int));
            _cached_ori_data_writer->write((char *)&data_dim_, sizeof(int));
            _cached_shard_data_writer.resize(_center_num);
            _cached_shard_idmap_writer.resize(_center_num);
            if (_use_half) {
                std::string ori_raw_filename = _partition_prefix + "_raw.data";
                _cached_raw_data_writer = new cached_ofstream(ori_raw_filename, BUFFER_SIZE_MID_FOR_CACHED_IO);
                _cached_raw_data_writer->write((char *)&_ori_data_num, sizeof(int));
                _cached_raw_data_writer->write((char *)&data_dim_, sizeof(int));
            }
            for (size_t i = 0; i < _center_num; i++)
            {
                std::string data_filename = _partition_prefix + "_subshard-" + std::to_string(i) + ".bin";
                std::string idmap_filename = _partition_prefix + "_subshard-" + std::to_string(i) + "_ids_uint32.bin";

                cached_ofstream *cached_shard_data_ofstream = new cached_ofstream(data_filename, BUFFER_SIZE_MID_FOR_CACHED_IO);
                _cached_shard_data_writer[i] = cached_shard_data_ofstream;
                _cached_shard_data_writer[i]->write((char *)&dummy_size, sizeof(uint32_t));
                _cached_shard_data_writer[i]->write((char *)&basedim32, sizeof(uint32_t));

                cached_ofstream *cached_shard_idmap_ofstream = new cached_ofstream(idmap_filename, BUFFER_SIZE_SMALL_FOR_CACHED_IO);
                _cached_shard_idmap_writer[i] = cached_shard_idmap_ofstream;
                _cached_shard_idmap_writer[i]->write((char *)&dummy_size, sizeof(uint32_t));
                _cached_shard_idmap_writer[i]->write((char *)&const_one, sizeof(uint32_t));
                _shard_counts[i] = 0;
            }
        } else {
            //init pq_code_profile
            _pq_table_populated = true;
            auto fragment_num = GetPqCentroidResource().getIntegrateMeta().fragmentNum;
            auto product_size = sizeof(uint16_t) * fragment_num;
            auto product_capacity = ArrayProfile::CalcSize(GetMaxDocNum(), product_size);
            pq_code_base_.assign(product_capacity, 0);
            if (!pq_code_profile_.create(pq_code_base_.data(), product_capacity, product_size)) {
                LOG_ERROR("Failed to create PQ code profile");
                return -1;
            }
        }
    }

    return 0;
}

int DiskVamanaIndex::BaseIndexAdd(docid_t doc_id, pk_t pk, const void *val, size_t len) 
{
    coarseVamanaIndex_->set_data_vec(doc_id, val);

    return 0;
}

int DiskVamanaIndex::AddOriData(const void *val)
{
    size_t dataLength = index_meta_.sizeofElement();

    _ori_data_num++;

    _cached_ori_data_writer->write((char *)(val), dataLength);

    return 0;
}

int DiskVamanaIndex::AddRawData(const void *val)
{
    size_t dataLength = index_meta_L2_.sizeofElement();

    _cached_raw_data_writer->write((char *)(val), dataLength);

    return 0;
}

int DiskVamanaIndex::AddShardData(int shard, docid_t doc_id, const void *val)
{
    size_t dataLength = index_meta_.sizeofElement();

    _cached_shard_data_writer[shard]->write((char *)(val), dataLength);

    _cached_shard_idmap_writer[shard]->write((char *)&doc_id, sizeof(doc_id));

    _shard_counts[shard]++;

    return 0;
}

int DiskVamanaIndex::PartitionBaseIndexAdd(docid_t doc_id, pk_t pk, const QueryInfo& query_info, const QueryInfo& query_info_raw) 
{
    const void *val = query_info.GetVector();
    size_t len = query_info.GetVectorLen();

    const void *raw = nullptr;
    size_t raw_len = 0;

    if (_use_half) {
        raw = query_info_raw.GetVector();
        raw_len = query_info_raw.GetVectorLen();
        _cached_raw_data_writer->write((char *)(raw), raw_len);
    }

    _ori_data_num++;
    _cached_ori_data_writer->write((char *)(val), len);

    std::priority_queue<CentroidInfo, std::vector<CentroidInfo>, std::greater<CentroidInfo>> centroids;

    for (uint32_t i = 0; i < _center_num; ++i) {
        const void *centroidValue = GetVamanaCentroidResource().getValueInRoughMatrix(0, i);
        float score = 0.0f;
        if (_use_half) {
            score = index_meta_L2_.distance(raw, centroidValue);
        } else {
            score = index_meta_L2_.distance(val, centroidValue);
        }
        centroids.emplace(i, score);
    }

    for (uint32_t k = 0; k < _k_base; k++) {
        if (centroids.empty()) {
            LOG_ERROR("max_shard_data_num should be greater than [2*total/(shard-1)]");
            std::abort();
        }
        uint32_t index = centroids.top().index;
        while (_shard_counts[index] >= max_shard_data_num_) {
            centroids.pop();
            if (centroids.empty()) {
                LOG_ERROR("max_shard_data_num should be greater than [2*total/(shard-1)]");
                std::abort();
            }
            index = centroids.top().index;
        }
        _cached_shard_data_writer[index]->write((char *)(val), len);
        _cached_shard_idmap_writer[index]->write((char *)&doc_id, sizeof(doc_id));
        centroids.pop();
        _shard_counts[index]++;
    }

    return 0;
}

int DiskVamanaIndex::PartitionBaseIndexRandAdd(docid_t doc_id, pk_t pk, const QueryInfo& query_info, const QueryInfo& query_info_raw) 
{
    const void *val = query_info.GetVector();
    size_t len = query_info.GetVectorLen();

    const void *raw = nullptr;
    size_t raw_len = 0;

    if (_use_half) {
        raw = query_info_raw.GetVector();
        raw_len = query_info_raw.GetVectorLen();
        _cached_raw_data_writer->write((char *)(raw), raw_len);
    }

    _ori_data_num++;
    _cached_ori_data_writer->write((char *)(val), len);

    std::unordered_set<uint32_t> indices;
    for (uint32_t k = 0; k < _k_base; k++) {
        uint32_t index = rand() % _center_num;
        while (indices.find(index) != indices.end()) {
            index = rand() % _center_num;
        }
        indices.insert(index);
        _cached_shard_data_writer[index]->write((char *)(val), len);
        _cached_shard_idmap_writer[index]->write((char *)&doc_id, sizeof(doc_id));
        _shard_counts[index]++;
    }

    return 0;
}

int DiskVamanaIndex::PartitionBaseIndexDump()
{
    _cached_ori_data_writer->reset();
    _cached_ori_data_writer->write((char *)&_ori_data_num, sizeof(int));
    _cached_ori_data_writer->write((char *)&data_dim_, sizeof(int));
    delete _cached_ori_data_writer;
    if (_use_half) {
        _cached_raw_data_writer->reset();
        _cached_raw_data_writer->write((char *)&_ori_data_num, sizeof(int));
        _cached_raw_data_writer->write((char *)&data_dim_, sizeof(int));
        delete _cached_raw_data_writer;
    }
    size_t total_count = 0;
    for (size_t i = 0; i < _center_num; i++)
    {
        uint32_t cur_shard_count = (uint32_t)_shard_counts[i];
        LOG_INFO("shard size is %d; ", cur_shard_count);
        total_count += cur_shard_count;
        _cached_shard_data_writer[i]->reset();
        _cached_shard_data_writer[i]->write((char *)&cur_shard_count, sizeof(uint32_t));
        delete _cached_shard_data_writer[i];
        _cached_shard_idmap_writer[i]->reset();
        _cached_shard_idmap_writer[i]->write((char *)&cur_shard_count, sizeof(uint32_t));
        delete _cached_shard_idmap_writer[i];
    }
    std::cout << "\n Partitioned " << _ori_data_num << " with replication factor " << _k_base << " to get "
                  << total_count << " points across " << _center_num << " shards " << std::endl;
    return 0;
}

int DiskVamanaIndex::ShardDataDump(std::string &shardToken, std::string &path)
{
    if (shardToken == "pq") {
        _cached_ori_data_writer->reset();
        _cached_ori_data_writer->write((char *)&_ori_data_num, sizeof(int));
        _cached_ori_data_writer->write((char *)&data_dim_, sizeof(int));
        if (_use_half) {
            _cached_raw_data_writer->reset();
            _cached_raw_data_writer->write((char *)&_ori_data_num, sizeof(int));
            _cached_raw_data_writer->write((char *)&data_dim_, sizeof(int));
        }
        path = _partition_prefix + "_ori.data";
    } else {
        size_t shard_id = std::stol(shardToken);
        uint32_t cur_shard_count = (uint32_t)_shard_counts[shard_id];
        LOG_INFO("shard size is %d; ", cur_shard_count);
        _cached_shard_data_writer[shard_id]->reset();
        _cached_shard_data_writer[shard_id]->write((char *)&cur_shard_count, sizeof(uint32_t));
        _cached_shard_idmap_writer[shard_id]->reset();
        _cached_shard_idmap_writer[shard_id]->write((char *)&cur_shard_count, sizeof(uint32_t));
        path = _partition_prefix + "_subshard-" + shardToken + "_ids_uint32.bin";
    }
    delete _cached_ori_data_writer;
    if (_use_half) {
        delete _cached_raw_data_writer;
    }
    for (size_t i = 0; i < _center_num; i++)
    {
        delete _cached_shard_data_writer[i];
        delete _cached_shard_idmap_writer[i];
    }
    return 0;
}

int DiskVamanaIndex::ShardIndexBuildAndDump(std::string &shardToken, std::string &indexPath)
{
    if (shardToken == "pq") {
        std::string pq_data_path = _partition_prefix + "_ori.data";
        if (_use_half) {
            pq_data_path = _partition_prefix + "_raw.data";
        }
        BuildPqIndexFromFile(pq_data_path);
        std::string vamana_partition_pq_index_path = _partition_prefix + "_pq.index";
        DumpPqLocal(vamana_partition_pq_index_path);
        indexPath = vamana_partition_pq_index_path;
    } else {
        size_t shard_id = std::stol(shardToken);
        std::string data_filename = _partition_prefix + "_subshard-" + std::to_string(shard_id) + ".bin";
        std::string index_filename = _partition_prefix + "_subshard-" + std::to_string(shard_id) + "_mem.index";
        size_t max_reserve_degree = (size_t)(defaults::GRAPH_SLACK_FACTOR * 1.05 * (2 * R_ / 3));
        
        uint64_t shard_base_dim, shard_base_pts;
        get_bin_metadata(data_filename, shard_base_pts, shard_base_dim);

        auto low_degree_params = IndexWriteParametersBuilder(L_, 2 * R_ / 3)
                                    .with_alpha(alpha_)
                                    .with_saturate_graph(is_saturated_)
                                    .with_max_occlusion_size(max_occlusion_)
                                    .with_num_threads(T_)
                                    .build();

        auto config = IndexConfigBuilder()
                        .with_dimension(shard_base_dim)
                        .with_max_points(shard_base_pts)
                        .with_max_reserve_degree(max_reserve_degree)
                        .is_dynamic_index(false)
                        .with_index_write_params(low_degree_params)
                        .is_enable_tags(false)
                        .build();

        _method = index_meta_.method();
        _measure = IndexDistance::EmbodyMeasure(_method);

        coarseVamanaIndex_ = std::make_unique<CoarseVamanaIndex>(data_size_, config, index_meta_.method(), _use_half);
        coarseVamanaIndex_->build(data_filename.c_str(), shard_base_pts);
        coarseVamanaIndex_->save(index_filename.c_str());
        coarseVamanaIndex_->_data_store.reset();
        coarseVamanaIndex_->_graph_store.reset();
        coarseVamanaIndex_.reset();
        std::remove(data_filename.c_str());
        indexPath = index_filename;
    }
    return 0;
}

void DiskVamanaIndex::BuildPqIndexFromFile(std::string filename)
{
    LOG_INFO("Start to build Pq Index");
    if(!file_exists(filename)) {
        LOG_ERROR("original data file not exists");
        throw std::io_errc();
    }
    //init pq_code_profile
    auto fragment_num = GetPqCentroidResource().getIntegrateMeta().fragmentNum;
    auto product_size = sizeof(uint16_t) * fragment_num;
    auto product_capacity = ArrayProfile::CalcSize(GetMaxDocNum(), product_size);
    pq_code_base_.assign(product_capacity, 0);
    if (!pq_code_profile_.create(pq_code_base_.data(), product_capacity, product_size)) {
        LOG_ERROR("Failed to init PQ code profile");
        throw std::runtime_error("Failed to init PQ code profile");
    }
    cached_ifstream reader;
    reader.open(filename, BUFFER_SIZE_LARGE_FOR_CACHED_IO);
    uint32_t data_num;
    uint32_t data_dim;
    reader.read((char *)&data_num, sizeof(uint32_t));
    reader.read((char *)&data_dim, sizeof(uint32_t));
    uint16_t data_size = index_meta_L2_.sizeofElement() / data_dim;
    for (uint32_t i = 0; i < data_num; i++) {
        std::unique_ptr<char[]> val = std::make_unique<char[]>(data_dim * data_size);
        reader.read((char *)val.get(), data_size * data_dim);
        PQIndexAdd(i, val.get());
    }
    _pq_table_populated = true;
    LOG_INFO("Build Pq Index done");
}

int DiskVamanaIndex::PQIndexAdd(docid_t doc_id, const void *val)
{
    QueryDistanceMatrix qdm(index_meta_L2_, &GetPqCentroidResource());
    bool bres = qdm.initDistanceMatrix(val, true);
    if (!bres) {
        LOG_ERROR("Calcualte QDM failed!");
        return -1;
    }
    
    std::vector<uint16_t> product_labels;
    if (!qdm.getQueryCodeFeature(product_labels)) {
        LOG_ERROR("get query codefeature failed!");
        return -1;
    }

    if (!pq_code_profile_.insert(doc_id, product_labels.data())) {
        LOG_ERROR("Failed to add into pq code profile.");
        return -1;
    }

    return 0;
}

void DiskVamanaIndex::BuildMemIndex()
{
    coarseVamanaIndex_->build_with_data_populated();
}

void DiskVamanaIndex::BuildAndDumpPartitionIndex()
{
    for (size_t i = 0; i < _center_num; i++)
    {
        std::string data_filename = _partition_prefix + "_subshard-" + std::to_string(i) + ".bin";
        std::string index_filename = _partition_prefix + "_subshard-" + std::to_string(i) + "_mem.index";
        size_t max_reserve_degree = (size_t)(defaults::GRAPH_SLACK_FACTOR * 1.05 * (2 * R_ / 3));
        
        uint64_t shard_base_dim, shard_base_pts;
        get_bin_metadata(data_filename, shard_base_pts, shard_base_dim);

        auto low_degree_params = IndexWriteParametersBuilder(L_, 2 * R_ / 3)
                                    .with_alpha(alpha_)
                                    .with_saturate_graph(is_saturated_)
                                    .with_max_occlusion_size(max_occlusion_)
                                    .with_num_threads(T_)
                                    .build();

        auto config = IndexConfigBuilder()
                        .with_dimension(shard_base_dim)
                        .with_max_points(shard_base_pts)
                        .with_max_reserve_degree(max_reserve_degree)
                        .is_dynamic_index(false)
                        .with_index_write_params(low_degree_params)
                        .is_enable_tags(false)
                        .build();

        if (i == 0) {
            _method = index_meta_.method();
            _measure = IndexDistance::EmbodyMeasure(_method);
        }

        coarseVamanaIndex_ = std::make_unique<CoarseVamanaIndex>(data_size_, config, index_meta_.method(), _use_half);
        coarseVamanaIndex_->build(data_filename.c_str(), shard_base_pts);
        coarseVamanaIndex_->save(index_filename.c_str());
        coarseVamanaIndex_->_data_store.reset();
        coarseVamanaIndex_->_graph_store.reset();
        coarseVamanaIndex_.reset();
        std::remove(data_filename.c_str());
    }
}

void DiskVamanaIndex::MergeShardIndex(std::string &path_medoids, size_t* size_medoids, 
                                      const std::vector<std::string> &shardIndexFiles, 
                                      const std::vector<std::string> &shardIdmapFiles) {
    // Read ID maps
    std::vector<std::string> vamana_names(_center_num);
    std::vector<std::vector<uint32_t>> idmaps(_center_num);
    for (uint32_t shard = 0; shard < _center_num; shard++)
    {
        vamana_names[shard] = shardIndexFiles[shard];
        read_idmap(shardIdmapFiles[shard], idmaps[shard]);
    }

    // find max node id
    size_t nnodes = 0;
    size_t nelems = 0;
    for (auto &idmap : idmaps)
    {
        for (auto &id : idmap)
        {
            nnodes = std::max(nnodes, (size_t)id);
        }
        nelems += idmap.size();
    }
    nnodes++;
    std::cout << "# nodes: " << nnodes << ", max. degree: " << R_ << std::endl;

    // compute inverse map: node -> shards
    std::vector<std::pair<uint32_t, uint32_t>> node_shard;
    node_shard.reserve(nelems);
    for (size_t shard = 0; shard < _center_num; shard++)
    {
        std::cout << "Creating inverse map -- shard #" << shard << std::endl;
        for (size_t idx = 0; idx < idmaps[shard].size(); idx++)
        {
            size_t node_id = idmaps[shard][idx];
            node_shard.push_back(std::make_pair((uint32_t)node_id, (uint32_t)shard));
        }
    }
    std::sort(node_shard.begin(), node_shard.end(), [](const auto &left, const auto &right) {
        return left.first < right.first || (left.first == right.first && left.second < right.second);
    });
    std::cout << "Finished computing node -> shards map" << std::endl;

    // create cached vamana readers
    std::vector<cached_ifstream> vamana_readers(_center_num);
    for (size_t i = 0; i < _center_num; i++)
    {
        vamana_readers[i].open(vamana_names[i], BUFFER_SIZE_LARGE_FOR_CACHED_IO);
        size_t expected_file_size;
        vamana_readers[i].read((char *)&expected_file_size, sizeof(uint64_t));
    }

    size_t vamana_metadata_size =
        sizeof(uint64_t) + sizeof(uint32_t) + sizeof(uint32_t) + sizeof(uint64_t); // expected file size + max degree +
                                                                                   // medoid_id + frozen_point info
    
    std::string output_vamana = _partition_prefix + "_merged_mem.index";
    
    // create cached vamana writers
    cached_ofstream merged_vamana_writer(output_vamana, BUFFER_SIZE_LARGE_FOR_CACHED_IO);

    size_t merged_index_size = vamana_metadata_size; // we initialize the size of the merged index to
                                                     // the metadata size
    size_t merged_index_frozen = 0;
    merged_vamana_writer.write((char *)&merged_index_size,
                               sizeof(uint64_t)); // we will overwrite the index size at the end

    uint32_t output_width = R_;
    uint32_t max_input_width = 0;
    // read width from each vamana to advance buffer by sizeof(uint32_t) bytes
    for (auto &reader : vamana_readers)
    {
        uint32_t input_width;
        reader.read((char *)&input_width, sizeof(uint32_t));
        max_input_width = input_width > max_input_width ? input_width : max_input_width;
    }

    std::cout << "Max input width: " << max_input_width << ", output width: " << output_width << std::endl;

    merged_vamana_writer.write((char *)&output_width, sizeof(uint32_t));
    std::string medoids_file = _partition_prefix + "_medoids.bin";
    size_t medoids_file_size = 0;
    std::ofstream medoid_writer(medoids_file.c_str(), std::ios::binary);
    uint32_t nshards_u32 = _center_num;
    uint32_t one_val = 1;
    medoid_writer.write((char *)&nshards_u32, sizeof(uint32_t));
    medoids_file_size += sizeof(uint32_t);
    medoid_writer.write((char *)&one_val, sizeof(uint32_t));
    medoids_file_size += sizeof(uint32_t);

    uint64_t vamana_index_frozen = 0; // as of now the functionality to merge many overlapping vamana
                                      // indices is supported only for bulk indices without frozen point.
                                      // Hence the final index will also not have any frozen points.
    for (uint64_t shard = 0; shard < _center_num; shard++)
    {
        uint32_t medoid;
        // read medoid
        vamana_readers[shard].read((char *)&medoid, sizeof(uint32_t));
        vamana_readers[shard].read((char *)&vamana_index_frozen, sizeof(uint64_t));
        if (vamana_index_frozen == true) {
            throw std::runtime_error("vamana_index_frozen should be false in such case!");
        }
        // rename medoid
        medoid = idmaps[shard][medoid];

        LOG_INFO("shard[%lu] has medoid with id[%d]", shard, medoid);

        medoid_writer.write((char *)&medoid, sizeof(uint32_t));

        medoids_file_size += sizeof(uint32_t);
        // write renamed medoid
        if (shard == (_center_num - 1)) //--> uncomment if running hierarchical
            merged_vamana_writer.write((char *)&medoid, sizeof(uint32_t));
    }
    merged_vamana_writer.write((char *)&merged_index_frozen, sizeof(uint64_t));
    medoid_writer.close();

    if (get_file_size(medoids_file) != medoids_file_size) {
        LOG_ERROR("Merge vamana index failed with expected medoids file size = %lu and actual file size = %lu", 
                    medoids_file_size, get_file_size(medoids_file));
        exit(-1);
    } else {
        *size_medoids = medoids_file_size;
        path_medoids = medoids_file;
    }

    std::cout << "Starting merge" << std::endl;

    std::random_device rng;
    std::mt19937 urng(rng());

    std::vector<bool> nhood_set(nnodes, 0);
    std::vector<uint32_t> final_nhood;

    uint32_t nnbrs = 0, shard_nnbrs = 0;
    uint32_t cur_id = 0;

    for (const auto &id_shard : node_shard)
    {
        uint32_t node_id = id_shard.first;
        uint32_t shard_id = id_shard.second;
        if (cur_id < node_id)
        {
            std::shuffle(final_nhood.begin(), final_nhood.end(), urng);
            nnbrs = (uint32_t)(std::min)(final_nhood.size(), (uint64_t)R_);
            // write into merged ofstream
            merged_vamana_writer.write((char *)&nnbrs, sizeof(uint32_t));
            merged_vamana_writer.write((char *)final_nhood.data(), nnbrs * sizeof(uint32_t));
            merged_index_size += (sizeof(uint32_t) + nnbrs * sizeof(uint32_t));
            if (cur_id % 499999 == 1)
            {
                std::cout << "." << std::flush;
            }
            cur_id = node_id;
            nnbrs = 0;
            for (auto &p : final_nhood)
                nhood_set[p] = 0;
            final_nhood.clear();
        }
        // read from shard_id ifstream
        vamana_readers[shard_id].read((char *)&shard_nnbrs, sizeof(uint32_t));

        if (shard_nnbrs == 0)
        {
            std::cout << "WARNING: shard #" << shard_id << ", node_id " << node_id << " has 0 nbrs" << std::endl;
        }

        std::vector<uint32_t> shard_nhood(shard_nnbrs);
        if (shard_nnbrs > 0)
            vamana_readers[shard_id].read((char *)shard_nhood.data(), shard_nnbrs * sizeof(uint32_t));
        // rename nodes
        for (uint64_t j = 0; j < shard_nnbrs; j++)
        {
            if (nhood_set[idmaps[shard_id][shard_nhood[j]]] == 0)
            {
                nhood_set[idmaps[shard_id][shard_nhood[j]]] = 1;
                final_nhood.emplace_back(idmaps[shard_id][shard_nhood[j]]);
            }
        }
    }

    std::shuffle(final_nhood.begin(), final_nhood.end(), urng);
    nnbrs = (uint32_t)(std::min)(final_nhood.size(), (uint64_t)R_);
    // write into merged ofstream
    merged_vamana_writer.write((char *)&nnbrs, sizeof(uint32_t));
    if (nnbrs > 0)
    {
        merged_vamana_writer.write((char *)final_nhood.data(), nnbrs * sizeof(uint32_t));
    }
    merged_index_size += (sizeof(uint32_t) + nnbrs * sizeof(uint32_t));
    for (auto &p : final_nhood)
        nhood_set[p] = 0;
    final_nhood.clear();

    std::cout << "Expected size: " << merged_index_size << std::endl;

    merged_vamana_writer.reset();
    merged_vamana_writer.write((char *)&merged_index_size, sizeof(uint64_t));

    std::cout << "Finished merge" << std::endl;

    // delete tempFiles
    for (uint32_t p = 0; p < _center_num; p++)
    {
        std::string shard_index_file = shardIndexFiles[p];
        std::string shard_id_file = shardIdmapFiles[p];

        std::remove(shard_id_file.c_str());
        std::remove(shard_index_file.c_str());
    }
}

void DiskVamanaIndex::MergePartitionIndex(std::string &path_medoids, size_t* size_medoids)
{
    // Read ID maps
    std::vector<std::string> vamana_names(_center_num);
    std::vector<std::vector<uint32_t>> idmaps(_center_num);
    for (uint32_t shard = 0; shard < _center_num; shard++)
    {
        vamana_names[shard] = _partition_prefix + "_subshard-" + std::to_string(shard) + "_mem.index";
        read_idmap(_partition_prefix + "_subshard-" + std::to_string(shard) + "_ids_uint32.bin", idmaps[shard]);
    }

    // find max node id
    size_t nnodes = 0;
    size_t nelems = 0;
    for (auto &idmap : idmaps)
    {
        for (auto &id : idmap)
        {
            nnodes = std::max(nnodes, (size_t)id);
        }
        nelems += idmap.size();
    }
    nnodes++;
    std::cout << "# nodes: " << nnodes << ", max. degree: " << R_ << std::endl;

    // compute inverse map: node -> shards
    std::vector<std::pair<uint32_t, uint32_t>> node_shard;
    node_shard.reserve(nelems);
    for (size_t shard = 0; shard < _center_num; shard++)
    {
        std::cout << "Creating inverse map -- shard #" << shard << std::endl;
        for (size_t idx = 0; idx < idmaps[shard].size(); idx++)
        {
            size_t node_id = idmaps[shard][idx];
            node_shard.push_back(std::make_pair((uint32_t)node_id, (uint32_t)shard));
        }
    }
    std::sort(node_shard.begin(), node_shard.end(), [](const auto &left, const auto &right) {
        return left.first < right.first || (left.first == right.first && left.second < right.second);
    });
    std::cout << "Finished computing node -> shards map" << std::endl;

    // create cached vamana readers
    std::vector<cached_ifstream> vamana_readers(_center_num);
    for (size_t i = 0; i < _center_num; i++)
    {
        vamana_readers[i].open(vamana_names[i], BUFFER_SIZE_LARGE_FOR_CACHED_IO);
        size_t expected_file_size;
        vamana_readers[i].read((char *)&expected_file_size, sizeof(uint64_t));
    }

    size_t vamana_metadata_size =
        sizeof(uint64_t) + sizeof(uint32_t) + sizeof(uint32_t) + sizeof(uint64_t); // expected file size + max degree +
                                                                                   // medoid_id + frozen_point info
    
    std::string output_vamana = _partition_prefix + "_merged_mem.index";
    
    // create cached vamana writers
    cached_ofstream merged_vamana_writer(output_vamana, BUFFER_SIZE_LARGE_FOR_CACHED_IO);

    size_t merged_index_size = vamana_metadata_size; // we initialize the size of the merged index to
                                                     // the metadata size
    size_t merged_index_frozen = 0;
    merged_vamana_writer.write((char *)&merged_index_size,
                               sizeof(uint64_t)); // we will overwrite the index size at the end

    uint32_t output_width = R_;
    uint32_t max_input_width = 0;
    // read width from each vamana to advance buffer by sizeof(uint32_t) bytes
    for (auto &reader : vamana_readers)
    {
        uint32_t input_width;
        reader.read((char *)&input_width, sizeof(uint32_t));
        max_input_width = input_width > max_input_width ? input_width : max_input_width;
    }

    std::cout << "Max input width: " << max_input_width << ", output width: " << output_width << std::endl;

    merged_vamana_writer.write((char *)&output_width, sizeof(uint32_t));
    std::string medoids_file = _partition_prefix + "_medoids.bin";
    size_t medoids_file_size = 0;
    std::ofstream medoid_writer(medoids_file.c_str(), std::ios::binary);
    uint32_t nshards_u32 = _center_num;
    uint32_t one_val = 1;
    medoid_writer.write((char *)&nshards_u32, sizeof(uint32_t));
    medoids_file_size += sizeof(uint32_t);
    medoid_writer.write((char *)&one_val, sizeof(uint32_t));
    medoids_file_size += sizeof(uint32_t);

    uint64_t vamana_index_frozen = 0; // as of now the functionality to merge many overlapping vamana
                                      // indices is supported only for bulk indices without frozen point.
                                      // Hence the final index will also not have any frozen points.
    for (uint64_t shard = 0; shard < _center_num; shard++)
    {
        uint32_t medoid;
        // read medoid
        vamana_readers[shard].read((char *)&medoid, sizeof(uint32_t));
        vamana_readers[shard].read((char *)&vamana_index_frozen, sizeof(uint64_t));
        if (vamana_index_frozen == true) {
            throw std::runtime_error("vamana_index_frozen should be false in such case!");
        }
        // rename medoid
        medoid = idmaps[shard][medoid];

        LOG_INFO("shard[%lu] has medoid with id[%d]", shard, medoid);

        medoid_writer.write((char *)&medoid, sizeof(uint32_t));

        medoids_file_size += sizeof(uint32_t);
        // write renamed medoid
        if (shard == (_center_num - 1)) //--> uncomment if running hierarchical
            merged_vamana_writer.write((char *)&medoid, sizeof(uint32_t));
    }
    merged_vamana_writer.write((char *)&merged_index_frozen, sizeof(uint64_t));
    medoid_writer.close();

    if (get_file_size(medoids_file) != medoids_file_size) {
        LOG_ERROR("Merge vamana index failed with expected medoids file size = %lu and actual file size = %lu", 
                    medoids_file_size, get_file_size(medoids_file));
        exit(-1);
    } else {
        *size_medoids = medoids_file_size;
        path_medoids = medoids_file;
    }

    std::cout << "Starting merge" << std::endl;

    std::random_device rng;
    std::mt19937 urng(rng());

    std::vector<bool> nhood_set(nnodes, 0);
    std::vector<uint32_t> final_nhood;

    uint32_t nnbrs = 0, shard_nnbrs = 0;
    uint32_t cur_id = 0;

    for (const auto &id_shard : node_shard)
    {
        uint32_t node_id = id_shard.first;
        uint32_t shard_id = id_shard.second;
        if (cur_id < node_id)
        {
            std::shuffle(final_nhood.begin(), final_nhood.end(), urng);
            nnbrs = (uint32_t)(std::min)(final_nhood.size(), (uint64_t)R_);
            // write into merged ofstream
            merged_vamana_writer.write((char *)&nnbrs, sizeof(uint32_t));
            merged_vamana_writer.write((char *)final_nhood.data(), nnbrs * sizeof(uint32_t));
            merged_index_size += (sizeof(uint32_t) + nnbrs * sizeof(uint32_t));
            if (cur_id % 499999 == 1)
            {
                std::cout << "." << std::flush;
            }
            cur_id = node_id;
            nnbrs = 0;
            for (auto &p : final_nhood)
                nhood_set[p] = 0;
            final_nhood.clear();
        }
        // read from shard_id ifstream
        vamana_readers[shard_id].read((char *)&shard_nnbrs, sizeof(uint32_t));

        if (shard_nnbrs == 0)
        {
            std::cout << "WARNING: shard #" << shard_id << ", node_id " << node_id << " has 0 nbrs" << std::endl;
        }

        std::vector<uint32_t> shard_nhood(shard_nnbrs);
        if (shard_nnbrs > 0)
            vamana_readers[shard_id].read((char *)shard_nhood.data(), shard_nnbrs * sizeof(uint32_t));
        // rename nodes
        for (uint64_t j = 0; j < shard_nnbrs; j++)
        {
            if (nhood_set[idmaps[shard_id][shard_nhood[j]]] == 0)
            {
                nhood_set[idmaps[shard_id][shard_nhood[j]]] = 1;
                final_nhood.emplace_back(idmaps[shard_id][shard_nhood[j]]);
            }
        }
    }

    std::shuffle(final_nhood.begin(), final_nhood.end(), urng);
    nnbrs = (uint32_t)(std::min)(final_nhood.size(), (uint64_t)R_);
    // write into merged ofstream
    merged_vamana_writer.write((char *)&nnbrs, sizeof(uint32_t));
    if (nnbrs > 0)
    {
        merged_vamana_writer.write((char *)final_nhood.data(), nnbrs * sizeof(uint32_t));
    }
    merged_index_size += (sizeof(uint32_t) + nnbrs * sizeof(uint32_t));
    for (auto &p : final_nhood)
        nhood_set[p] = 0;
    final_nhood.clear();

    std::cout << "Expected size: " << merged_index_size << std::endl;

    merged_vamana_writer.reset();
    merged_vamana_writer.write((char *)&merged_index_size, sizeof(uint64_t));

    std::cout << "Finished merge" << std::endl;

    // delete tempFiles
    for (uint32_t p = 0; p < _center_num; p++)
    {
        std::string shard_id_file = _partition_prefix + "_subshard-" + std::to_string(p) + "_ids_uint32.bin";
        std::string shard_index_file = _partition_prefix + "_subshard-" + std::to_string(p) + "_mem.index";
        std::string shard_index_file_data = shard_index_file + ".data";

        std::remove(shard_id_file.c_str());
        std::remove(shard_index_file.c_str());
        std::remove(shard_index_file_data.c_str());
    }

}

void DiskVamanaIndex::DumpMemLocal(const std::string filename)
{
    coarseVamanaIndex_->save(filename.c_str());
}

int DiskVamanaIndex::Dump(const void*& data, size_t& size)
{
    // dump common
    if (DumpHelper::DumpCommon(this, index_package_) != 0) {
        LOG_ERROR("dump into package failed.");
        return -1;
    }

    // dump pq index
    GetPqCentroidResource().dumpIntegrateMatrix(integrate_matrix_);

    index_package_.emplace(COMPONENT_INTEGRATE_MATRIX, integrate_matrix_.data(), integrate_matrix_.size());

    size_t real_capacity = sizeof(mercury::core::ArrayProfile::Header) 
                            + (pq_code_profile_.getHeader()->infoSize * pq_code_profile_.getHeader()->usedDocNum);

    pq_code_profile_.setCapacity(real_capacity);

    index_package_.emplace(COMPONENT_PQ_CODE_PROFILE, pq_code_profile_.getHeader(), real_capacity);

    if (!index_package_.dump(data, size)) {
        LOG_ERROR("Failed to dump package.");
        return -1;
    }

    LOG_INFO("DiskVamanaIndex::Dump End! index_package_ size: %lu", size);

    return 0;
}

int DiskVamanaIndex::DumpPqLocal(const std::string& pq_file_name)
{
    LOG_INFO("Start to dump Pq Index");
    // dump common
    if (DumpHelper::DumpCommon(this, index_package_) != 0) {
        LOG_ERROR("dump into package failed.");
        return -1;
    }

    // dump pq index
    GetPqCentroidResource().dumpIntegrateMatrix(integrate_matrix_);

    index_package_.emplace(COMPONENT_INTEGRATE_MATRIX, integrate_matrix_.data(), integrate_matrix_.size());

    size_t real_capacity = sizeof(mercury::core::ArrayProfile::Header) 
                            + (pq_code_profile_.getHeader()->infoSize * pq_code_profile_.getHeader()->usedDocNum);

    pq_code_profile_.setCapacity(real_capacity);

    index_package_.emplace(COMPONENT_PQ_CODE_PROFILE, pq_code_profile_.getHeader(), real_capacity);

    auto stg = InstanceFactory::CreateStorage("MMapFileStorage");
    if (!index_package_.dump(pq_file_name, stg)) {
        LOG_ERROR("Failed to dump package.");
        return -1;
    }

    pq_code_base_.clear();

    pq_code_base_.shrink_to_fit();

    LOG_INFO("Dump Pq Index done");

    return 0;

}

void DiskVamanaIndex::CreateDiskLayout(const std::string base_file, const std::string mem_index_file, const std::string output_file)
{
    uint32_t npts, ndims;

    // amount to read or write in one shot
    size_t read_blk_size = 64 * 1024 * 1024;
    size_t write_blk_size = read_blk_size;
    cached_ifstream base_reader(base_file, read_blk_size);
    base_reader.read((char *)&npts, sizeof(uint32_t));
    base_reader.read((char *)&ndims, sizeof(uint32_t));

    size_t npts_64, ndims_64;
    npts_64 = npts;
    ndims_64 = ndims;

    LOG_INFO("npts_64: %lu", npts_64);
    LOG_INFO("ndims_64: %lu", ndims_64);

    // Check if we need to append data for re-ordering
    bool append_reorder_data = false;
    std::ifstream reorder_data_reader;

    uint32_t npts_reorder_file = 0, ndims_reorder_file = 0;
    // create cached reader + writer
    size_t actual_file_size = get_file_size(mem_index_file);
    LOG_INFO("Vamana index file size=%lu", actual_file_size);
    std::ifstream vamana_reader(mem_index_file, std::ios::binary);
    cached_ofstream diskann_writer(output_file, write_blk_size);

    // metadata: width, medoid
    uint32_t width_u32, medoid_u32;
    size_t index_file_size;

    vamana_reader.read((char *)&index_file_size, sizeof(uint64_t));
    if (index_file_size != actual_file_size)
    {
        std::stringstream stream;
        stream << "Vamana Index file size does not match expected size per "
                  "meta-data."
               << " file size from file: " << index_file_size << " actual file size: " << actual_file_size << std::endl;

        throw std::runtime_error(stream.str());
    }
    uint64_t vamana_frozen_num = false, vamana_frozen_loc = 0;

    vamana_reader.read((char *)&width_u32, sizeof(uint32_t));
    vamana_reader.read((char *)&medoid_u32, sizeof(uint32_t));
    vamana_reader.read((char *)&vamana_frozen_num, sizeof(uint64_t));
    // compute
    uint64_t medoid, max_node_len, nnodes_per_sector;
    medoid = (uint64_t)medoid_u32;
    if (vamana_frozen_num == 1)
        vamana_frozen_loc = medoid;
    max_node_len = (((uint64_t)width_u32 + 1) * sizeof(uint32_t)) + (ndims_64 * data_size_);
    nnodes_per_sector = defaults::SECTOR_LEN / max_node_len; // 0 if max_node_len > SECTOR_LEN

    LOG_INFO("medoid: %luB", medoid);
    LOG_INFO("max_node_len: %luB", max_node_len);
    LOG_INFO("nnodes_per_sector: %luB", nnodes_per_sector);

    // defaults::SECTOR_LEN buffer for each sector
    std::unique_ptr<char[]> sector_buf = std::make_unique<char[]>(defaults::SECTOR_LEN);
    std::unique_ptr<char[]> multisector_buf = std::make_unique<char[]>(ROUND_UP(max_node_len, defaults::SECTOR_LEN));
    std::unique_ptr<char[]> node_buf = std::make_unique<char[]>(max_node_len);
    uint32_t &nnbrs = *(uint32_t *)(node_buf.get() + ndims_64 * data_size_);
    uint32_t *nhood_buf = (uint32_t *)(node_buf.get() + (ndims_64 * data_size_) + sizeof(uint32_t));

    // number of sectors (1 for meta data)
    uint64_t n_sectors = nnodes_per_sector > 0 ? ROUND_UP(npts_64, nnodes_per_sector) / nnodes_per_sector
                                               : npts_64 * DIV_ROUND_UP(max_node_len, defaults::SECTOR_LEN);
    uint64_t n_reorder_sectors = 0;
    uint64_t n_data_nodes_per_sector = 0;

    uint64_t disk_index_file_size = (n_sectors + n_reorder_sectors + 1) * defaults::SECTOR_LEN;

    std::vector<uint64_t> output_file_meta;
    output_file_meta.push_back(npts_64);
    output_file_meta.push_back(ndims_64);
    output_file_meta.push_back(medoid);
    output_file_meta.push_back(max_node_len);
    output_file_meta.push_back(nnodes_per_sector);
    output_file_meta.push_back(vamana_frozen_num);
    output_file_meta.push_back(vamana_frozen_loc);
    output_file_meta.push_back((uint64_t)append_reorder_data);
    output_file_meta.push_back(disk_index_file_size);

    diskann_writer.write(sector_buf.get(), defaults::SECTOR_LEN);

    std::unique_ptr<char[]> cur_node_coords = std::make_unique<char[]>(ndims_64 * data_size_);

    LOG_INFO("# sectors: %lu", n_sectors);
    uint64_t cur_node_id = 0;

    if (nnodes_per_sector > 0)
    { // Write multiple nodes per sector
        for (uint64_t sector = 0; sector < n_sectors; sector++)
        {
            if (sector % 100000 == 0)
            {
                LOG_INFO("Sector #%lu written", sector);
            }
            memset(sector_buf.get(), 0, defaults::SECTOR_LEN);
            for (uint64_t sector_node_id = 0; sector_node_id < nnodes_per_sector && cur_node_id < npts_64;
                 sector_node_id++)
            {
                memset(node_buf.get(), 0, max_node_len);
                // read cur node's nnbrs
                vamana_reader.read((char *)&nnbrs, sizeof(uint32_t));

                // sanity checks on nnbrs
                assert(nnbrs > 0);
                assert(nnbrs <= width_u32);

                // read node's nhood
                vamana_reader.read((char *)nhood_buf, (std::min)(nnbrs, width_u32) * sizeof(uint32_t));
                if (nnbrs > width_u32)
                {
                    vamana_reader.seekg((nnbrs - width_u32) * sizeof(uint32_t), vamana_reader.cur);
                }

                // write coords of node first
                base_reader.read((char *)cur_node_coords.get(), data_size_ * ndims_64);
                memcpy(node_buf.get(), cur_node_coords.get(), ndims_64 * data_size_);

                // write nnbrs
                *(uint32_t *)(node_buf.get() + ndims_64 * data_size_) = (std::min)(nnbrs, width_u32);

                // write nhood next
                memcpy(node_buf.get() + ndims_64 * data_size_ + sizeof(uint32_t), nhood_buf,
                       (std::min)(nnbrs, width_u32) * sizeof(uint32_t));

                // get offset into sector_buf
                char *sector_node_buf = sector_buf.get() + (sector_node_id * max_node_len);

                // copy node buf into sector_node_buf
                memcpy(sector_node_buf, node_buf.get(), max_node_len);
                cur_node_id++;
            }
            // flush sector to disk
            diskann_writer.write(sector_buf.get(), defaults::SECTOR_LEN);
        }
    }
    else
    { // Write multi-sector nodes
        uint64_t nsectors_per_node = DIV_ROUND_UP(max_node_len, defaults::SECTOR_LEN);
        for (uint64_t i = 0; i < npts_64; i++)
        {
            if ((i * nsectors_per_node) % 100000 == 0)
            {
                LOG_INFO("Sector #%lu written", i * nsectors_per_node);
            }
            memset(multisector_buf.get(), 0, nsectors_per_node * defaults::SECTOR_LEN);

            memset(node_buf.get(), 0, max_node_len);
            // read cur node's nnbrs
            vamana_reader.read((char *)&nnbrs, sizeof(uint32_t));

            // sanity checks on nnbrs
            assert(nnbrs > 0);
            assert(nnbrs <= width_u32);

            // read node's nhood
            vamana_reader.read((char *)nhood_buf, (std::min)(nnbrs, width_u32) * sizeof(uint32_t));
            if (nnbrs > width_u32)
            {
                vamana_reader.seekg((nnbrs - width_u32) * sizeof(uint32_t), vamana_reader.cur);
            }

            // write coords of node first
            base_reader.read((char *)cur_node_coords.get(), data_size_ * ndims_64);
            memcpy(node_buf.get(), cur_node_coords.get(), ndims_64 * data_size_);

            // write nnbrs
            *(uint32_t *)(multisector_buf.get() + ndims_64 * data_size_) = (std::min)(nnbrs, width_u32);

            // write nhood next
            memcpy(multisector_buf.get() + ndims_64 * data_size_ + sizeof(uint32_t), nhood_buf,
                   (std::min)(nnbrs, width_u32) * sizeof(uint32_t));

            // flush sector to disk
            diskann_writer.write(multisector_buf.get(), nsectors_per_node * defaults::SECTOR_LEN);
        }
    }

    diskann_writer.close();

    save_bin<uint64_t>(output_file, output_file_meta.data(), output_file_meta.size(), 1, 0);
    LOG_INFO("Output disk index file written to %s", output_file.c_str());
}

float DiskVamanaIndex::compare(const void *a, const void *b, uint32_t length) {
    return _measure(a, b, length * data_size_);
}

bool DiskVamanaIndex::InitPqCentroidMatrix(const IndexParams& param) {
    uint16_t pq_centroid_num = GetOrDefault(param, PARAM_PQ_CENTROID_NUM, DefaultPqCentroidNum);
    uint16_t pq_fragment_count = GetOrDefault(param, PARAM_PQ_FRAGMENT_NUM, DefaultPqFragmentCnt);

    CentroidResource centroid_resource;

    //init pq
    CentroidResource::IntegrateMeta integrate_meta(index_meta_L2_.sizeofElement() / pq_fragment_count,
                                                   pq_fragment_count, pq_centroid_num);

    if (!centroid_resource.create(integrate_meta)) {
        LOG_ERROR("Failed to create integrate meta.");
        return false;
    }

    centroid_resource_manager_.AddCentroidResource(std::move(centroid_resource));

    size_t pq_element_size = index_meta_L2_.sizeofElement() / pq_fragment_count;
    //pq, 遍历所有分片
    for (int i = 0; i < pq_fragment_count; i++) {
        size_t centroid_size = 0;
        std::string file_path = param.getString(PARAM_TRAIN_DATA_PATH) + PQ_CENTROID_FILE_MIDDLEFIX
            + std::to_string(i) + PQ_CENTROID_FILE_POSTFIX;
        MatrixPointer matrix_pointer = DoLoadCentroidMatrix(file_path,
                                                            index_meta_L2_.dimension() / pq_fragment_count,
                                                            pq_element_size,
                                                            centroid_size);
        if (!matrix_pointer) {
            LOG_ERROR("Failed to Load centroid Matrix.");
            return false;
        }
        char* centroid_matrix = matrix_pointer.get();

        size_t j = 0;
        for (; j < centroid_size; j++) {
            if (!GetPqCentroidResource().setValueInIntegrateMatrix(i, j, centroid_matrix + j * pq_element_size)) {
                LOG_ERROR("Failed to set centroid resource rough matrix.");
                return false;
            }
        }

        //if not enough pq centroids, use last to make up
        for (; j < pq_centroid_num; j++) {
            if (!GetPqCentroidResource().setValueInIntegrateMatrix(i, j, centroid_matrix + (centroid_size - 1) * pq_element_size)) {
                LOG_ERROR("Failed to make up centroid resource rough matrix.");
                return false;
            }
        }
    }

    return true;
}

bool DiskVamanaIndex::InitVamanaCentroidMatrix(const std::string& centroid_dir) {

    std::string centroid_path = centroid_dir + "/" + "0-0" + IVF_CENTROID_FILE_POSTFIX;

    size_t centroid_size = 0;

    MatrixPointer matrix_pointer = DoLoadCentroidMatrix(centroid_path, index_meta_L2_.dimension(),
                                                            index_meta_L2_.sizeofElement(), centroid_size);
    if (!matrix_pointer) {
        LOG_ERROR("Failed to Load centroid Matrix.");
        return false;
    }

    char* centroid_matrix = matrix_pointer.get();
    std::vector<uint32_t> centroids_levelcnts;
    centroids_levelcnts.push_back(centroid_size);

    CentroidResource centroid_resource;
    //only support one level cnt, DefaultLevelCnt = 1.
    CentroidResource::RoughMeta rough_meta(index_meta_L2_.sizeofElement(), DefaultLevelCnt, centroids_levelcnts);
    if (!centroid_resource.create(rough_meta)) {
        LOG_ERROR("Failed to create centroid resource.");
        return false;
    }

    centroid_resource_manager_.AddCentroidResource(std::move(centroid_resource));

    //only 1 level
    for (size_t i = 0; i < centroid_size; i++) {
        if (!GetVamanaCentroidResource().setValueInRoughMatrix(0, i, centroid_matrix + i * index_meta_L2_.sizeofElement())) {
            LOG_ERROR("Failed to set centroid resource rough matrix.");
            return false;
        }
    }

    return true;
}

MatrixPointer DiskVamanaIndex::DoLoadCentroidMatrix(const std::string& file_path, size_t dimension,
                                                    size_t element_size, size_t& centroid_size) const {
    std::string file_content;
    if (!HdfsFileWrapper::AtomicLoad(file_path, file_content)) {
        LOG_ERROR("load centroid file failed.");
        return NullMatrix();
    }

    std::vector<std::string> centroid_vec = StringUtil::split(file_content, "\n");
    centroid_size = centroid_vec.size();
    char* centroid_matrix = new char[element_size * centroid_vec.size()];
    if (centroid_matrix == nullptr) {
        LOG_ERROR("centroid_matrix is null");
        return NullMatrix();
    }

    MatrixPointer matrix_pointer(centroid_matrix);

    //loop every centroid
    for (size_t i = 0 ; i < centroid_vec.size(); i++) {
        std::vector<std::string> centroid_item = StringUtil::split(centroid_vec.at(i), " ");
        if (centroid_item.size() != 2) {
            LOG_ERROR("centroid_item space split format error: %s", centroid_vec.at(i).c_str());
            return NullMatrix();
        }

        std::vector<string> values = StringUtil::split(centroid_item.at(1), ",");
        if (values.size() != dimension) {
            LOG_ERROR("centroid value commar split format error: %s", centroid_item.at(1).c_str());
            LOG_ERROR("values size: %lu and dimension: %lu", values.size(), dimension);
            return NullMatrix();
        }

        //loop every dimension
        for (size_t j = 0; j < values.size(); j++) {
            char value[32];//here at most int64_t 8 Byte, 32 is enough
            if(!StrToValue(values.at(j), (void*)value)) {
                LOG_ERROR("centroid value str to value format error: %s", values.at(j).c_str());
                return NullMatrix();
            }

            size_t type_size = element_size / dimension;
            memcpy((char*)centroid_matrix + element_size * i + j * type_size,
                   (void*)value, type_size);
        }
    }

    return std::move(matrix_pointer);
}

void DiskVamanaIndex::cache_bfs_levels(uint64_t num_nodes_to_cache, std::vector<uint32_t> &node_list) {

    tsl::robin_set<uint32_t> node_set;

    // Do not cache more than 10% of the nodes in the index
    uint64_t tenp_nodes = (uint64_t)(std::round(this->doc_num_ * 0.1));
    if (num_nodes_to_cache > tenp_nodes)
    {
        LOG_INFO("Reducing nodes to cache from: %lu to: %lu (10 percent of total nodes: %lu)", num_nodes_to_cache, tenp_nodes, this->doc_num_);
        num_nodes_to_cache = tenp_nodes == 0 ? 1 : tenp_nodes;
    }
    LOG_INFO("Caching %lu ...", num_nodes_to_cache);

    // borrow ctx
    ScratchStoreManager<SSDThreadData> manager(this->_thread_data);
    auto data = manager.scratch_space();
    IOContext &ctx = data->ctx;

    std::unique_ptr<tsl::robin_set<uint32_t>> cur_level, prev_level;
    cur_level = std::make_unique<tsl::robin_set<uint32_t>>();
    prev_level = std::make_unique<tsl::robin_set<uint32_t>>();

    for (uint64_t miter = 0; miter < _num_medoids && cur_level->size() < num_nodes_to_cache; miter++)
    {
        cur_level->insert(_medoids[miter]);
    }

    uint64_t lvl = 1;
    uint64_t prev_node_set_size = 0;
    while ((node_set.size() + cur_level->size() < num_nodes_to_cache) && cur_level->size() != 0)
    {
        // swap prev_level and cur_level
        std::swap(prev_level, cur_level);
        // clear cur_level
        cur_level->clear();

        std::vector<uint32_t> nodes_to_expand;

        for (const uint32_t &id : *prev_level)
        {
            if (node_set.find(id) != node_set.end())
            {
                continue;
            }
            node_set.insert(id);
            nodes_to_expand.push_back(id);
        }

        std::sort(nodes_to_expand.begin(), nodes_to_expand.end());

        LOG_INFO("Level: %lu", lvl);
        bool finish_flag = false;

        uint64_t BLOCK_SIZE = defaults::CACHE_LEVEL_BLOCK_SIZE;
        uint64_t nblocks = DIV_ROUND_UP(nodes_to_expand.size(), BLOCK_SIZE);
        for (size_t block = 0; block < nblocks && !finish_flag; block++)
        {
            LOG_INFO(".");
            size_t start = block * BLOCK_SIZE;
            size_t end = (std::min)((block + 1) * BLOCK_SIZE, nodes_to_expand.size());

            std::vector<uint32_t> nodes_to_read;
            std::vector<float *> coord_buffers(end - start, nullptr);
            std::vector<half_float::half *> coord_buffers_half(end - start, nullptr);
            std::vector<std::pair<uint32_t, uint32_t *>> nbr_buffers;

            for (size_t cur_pt = start; cur_pt < end; cur_pt++)
            {
                nodes_to_read.push_back(nodes_to_expand[cur_pt]);
                nbr_buffers.emplace_back(0, new uint32_t[_max_degree + 1]);
            }

            // issue read requests
            auto read_status = read_nodes(nodes_to_read, coord_buffers, coord_buffers_half, nbr_buffers);

            // process each nhood buf
            for (uint32_t i = 0; i < read_status.size(); i++)
            {
                if (read_status[i] == false)
                {
                    continue;
                }
                else
                {
                    uint32_t nnbrs = nbr_buffers[i].first;
                    uint32_t *nbrs = nbr_buffers[i].second;

                    // explore next level
                    for (uint32_t j = 0; j < nnbrs && !finish_flag; j++)
                    {
                        if (node_set.find(nbrs[j]) == node_set.end())
                        {
                            cur_level->insert(nbrs[j]);
                        }
                        if (cur_level->size() + node_set.size() >= num_nodes_to_cache)
                        {
                            finish_flag = true;
                        }
                    }
                }
                delete[] nbr_buffers[i].second;
            }
        }

        LOG_INFO(". #nodes: %lu, #nodes thus far: %lu", node_set.size() - prev_node_set_size, node_set.size());

        prev_node_set_size = node_set.size();
        lvl++;
    }

    if (node_set.size() + cur_level->size() != num_nodes_to_cache && cur_level->size() != 0) {
        throw std::runtime_error("cache node num not as expected!!!");
    }

    node_list.clear();
    node_list.reserve(node_set.size() + cur_level->size());
    for (auto node : node_set)
        node_list.push_back(node);
    for (auto node : *cur_level)
        node_list.push_back(node);

    LOG_INFO("Level: %lu", lvl);
    LOG_INFO(". #nodes: %lu, #nodes thus far: %lu", node_list.size() - prev_node_set_size, node_list.size());
    LOG_INFO("done");
}

void DiskVamanaIndex::load_cache_list(std::vector<uint32_t> &node_list) {
    LOG_INFO("Loading the cache list into memory..");
    
    // borrow ctx
    ScratchStoreManager<SSDThreadData> manager(this->_thread_data);
    auto data = manager.scratch_space();
    IOContext &ctx = data->ctx;

    size_t num_cached_nodes = node_list.size();

    LOG_INFO("Reserve space for robin_map");

    if (_use_half) {
        coord_cache_half_ = std::make_unique<tsl::robin_map<uint32_t, half_float::half *>>(num_cached_nodes);
    } else {
        coord_cache_ = std::make_unique<tsl::robin_map<uint32_t, float *>>(num_cached_nodes);
    }
    nhood_cache_ = std::make_unique<tsl::robin_map<uint32_t, std::pair<uint32_t, uint32_t *>>>(num_cached_nodes);

    // Allocate space for neighborhood cache
    _nhood_cache_buf = new uint32_t[num_cached_nodes * (_max_degree + 1)];
    memset(_nhood_cache_buf, 0, num_cached_nodes * (_max_degree + 1));

    // Allocate space for coordinate cache
    size_t coord_cache_buf_len = num_cached_nodes * aligned_dim_;
    if (_use_half) {
        _coord_cache_buf_half.resize(coord_cache_buf_len);
    } else {
        _coord_cache_buf.resize(coord_cache_buf_len);
    }

    LOG_INFO("Alloc coord_cache_buf succeed..");

    size_t BLOCK_SIZE = defaults::CACHE_LOAD_BLOCK_SIZE;
    size_t num_blocks = DIV_ROUND_UP(num_cached_nodes, BLOCK_SIZE);
    LOG_INFO("num_blocks is %lu", num_blocks);
    for (size_t block = 0; block < num_blocks; block++)
    {
        size_t start_idx = block * BLOCK_SIZE;
        size_t end_idx = (std::min)(num_cached_nodes, (block + 1) * BLOCK_SIZE);

        // Copy offset into buffers to read into
        std::vector<uint32_t> nodes_to_read;
        std::vector<float *> coord_buffers;
        std::vector<half_float::half *> coord_buffers_half;
        std::vector<std::pair<uint32_t, uint32_t *>> nbr_buffers;
        for (size_t node_idx = start_idx; node_idx < end_idx; node_idx++)
        {
            nodes_to_read.push_back(node_list[node_idx]);
            if (_use_half) {
                coord_buffers_half.push_back(_coord_cache_buf_half.data() + node_idx * aligned_dim_);
            } else {
                coord_buffers.push_back(_coord_cache_buf.data() + node_idx * aligned_dim_);
            }
            nbr_buffers.emplace_back(0, _nhood_cache_buf + node_idx * (_max_degree + 1));
        }

        // issue the reads
        auto read_status = read_nodes(nodes_to_read, coord_buffers, coord_buffers_half, nbr_buffers);

        // check for success and insert into the cache.
        for (size_t i = 0; i < read_status.size(); i++)
        {
            if (read_status[i] == true)
            {
                if (_use_half) {
                    coord_cache_half_->insert(std::make_pair(nodes_to_read[i], coord_buffers_half[i]));
                } else {
                    coord_cache_->insert(std::make_pair(nodes_to_read[i], coord_buffers[i]));
                }
                nhood_cache_->insert(std::make_pair(nodes_to_read[i], nbr_buffers[i]));
            }
        }
    }
    LOG_INFO("load_cache_list done...");
}

int DiskVamanaIndex::Load(const void* data, size_t size) {
    LOG_INFO("Begin to Load disk_vamana_index. size: %lu.", size);
    if (Index::Load(data, size) != 0) {
        LOG_ERROR("Failed to call Index::Load.");
        return -1;
    }

    index_meta_L2_ = index_meta_;
    index_meta_L2_.setType(mercury::core::IndexMeta::FeatureTypes::kTypeFloat);
    index_meta_L2_.setMethod(mercury::core::IndexDistance::kMethodFloatSquaredEuclidean);
    

    auto *component = index_package_.get(COMPONENT_INTEGRATE_MATRIX);
    if (!component) {
        LOG_ERROR("get component error: %s", COMPONENT_INTEGRATE_MATRIX);
        return -1;
    }

    data_dim_ = index_meta_.dimension();

    data_size_ = index_meta_.sizeofElement() / index_meta_.dimension();

    _method = index_meta_.method();
    
    _measure = IndexDistance::EmbodyMeasure(_method);

    // determine data type
    if (index_meta_.type() == mercury::core::IndexMeta::kTypeHalfFloat) {
        _use_half = true;
    }

    uint16_t pq_centroid_num = DefaultPqCentroidNum;

    uint16_t pq_fragment_count = DefaultPqFragmentCnt;

    CentroidResource centroid_resource;

    //init pq
    CentroidResource::IntegrateMeta integrate_meta(index_meta_L2_.sizeofElement() / pq_fragment_count,
                                                   pq_fragment_count, pq_centroid_num);

    if (!centroid_resource.create(integrate_meta)) {
        LOG_ERROR("Failed to create integrate meta.");
        return false;
    }

    centroid_resource_manager_.AddCentroidResource(std::move(centroid_resource));
    
    if (!GetPqCentroidResource().initIntegrate((void *)component->getData(), component->getDataSize())) {
        LOG_ERROR("centroid resource integrate init error");
        return -1;
    }

    component = index_package_.get(COMPONENT_PQ_CODE_PROFILE);
    if (!component) {
        LOG_WARN("get component error : %s", COMPONENT_PQ_CODE_PROFILE);
    } else {
        if (!pq_code_profile_.load((void*)component->getData(), component->getDataSize())) {
            LOG_ERROR("pq code profile load error");
            return -1;
        }
    }
    return 0;
}

int DiskVamanaIndex::LoadDiskIndex(const std::string& disk_index_path, const std::string& medoids_data_path) {
    
    std::ifstream index_metadata(disk_index_path, std::ios::binary);

    uint32_t nr, nc; // metadata itself is stored as bin format (nr is number of
                    // metadata, nc should be 1)
    READ_INT(index_metadata, nr);
    READ_INT(index_metadata, nc);

    LOG_INFO("nr: %d", nr);

    LOG_INFO("nc: %d", nc);

    uint64_t disk_nnodes;
    uint64_t disk_ndims; // can be disk PQ dim if disk_PQ is set to true
    READ_U64(index_metadata, disk_nnodes);
    READ_U64(index_metadata, disk_ndims);

    doc_num_ = disk_nnodes;

    LOG_INFO("disk_nnodes: %lu", disk_nnodes);

    LOG_INFO("disk_ndims: %lu", disk_ndims);

    aligned_dim_ = disk_ndims;

    data_dim_ = disk_ndims;

    size_t medoid_id_on_file;
    READ_U64(index_metadata, medoid_id_on_file);
    LOG_INFO("medoid_id_on_file: %luB", medoid_id_on_file);

    
    READ_U64(index_metadata, _max_node_len);
    READ_U64(index_metadata, _nnodes_per_sector);
    LOG_INFO("_max_node_len: %luB", _max_node_len);
    LOG_INFO("_nnodes_per_sector: %luB", _nnodes_per_sector);

    _max_degree = ((_max_node_len - _disk_bytes_per_point) / sizeof(uint32_t)) - 1;
    LOG_INFO("_max_degree: %lu", _max_degree);

    READ_U64(index_metadata, this->_num_frozen_points);
    LOG_INFO("_num_frozen_points: %lu", _num_frozen_points);
    uint64_t file_frozen_id;
    READ_U64(index_metadata, file_frozen_id);
    LOG_INFO("file_frozen_id: %lu", file_frozen_id);

    READ_U64(index_metadata, this->_reorder_data_exists);
    LOG_INFO("_reorder_data_exists: %s", _reorder_data_exists ? "true" : "false");

    index_metadata.close();

    reader.reset(new AlignedFileReader());
    
    reader->open(disk_index_path);

    LOG_INFO("Check AIO MAX NR ...");

    if (stoi(shell_exec(defaults::CMD_GET_AIO_MAX_NR)) < defaults::AIO_MAX_NR) {
        std::string msg = shell_exec(defaults::CMD_SET_AIO_MAX_NR);
        if (stoi(shell_exec(defaults::CMD_GET_AIO_MAX_NR)) != defaults::AIO_MAX_NR) {
            LOG_ERROR("failed to set aio max nr with message: %s", msg.c_str());
            return -1;
        } else {
            LOG_INFO("succeed to set aio max nr with message: %s", msg.c_str());
        }
    }

    LOG_INFO("Begin to set aio context...");

    std::lock_guard<std::mutex> lock(g_io_mutex);

    for (uint32_t i = 0; i < defaults::IO_CONTEXT_NUM; i++) {
        SSDThreadData *data = new SSDThreadData(data_size_, disk_ndims);

        this->reader->register_thread(i);

        data->ctx = this->reader->get_ctx(i);

        _thread_data.push(data);
    }

    _disk_bytes_per_point = this->data_dim_ * data_size_;

    LOG_INFO("Begin to load medoids");

    if (medoids_data_path == "") {

        LOG_INFO("medoids file not exists, use default medoid");

        _num_medoids = 1;

        _medoids = new uint32_t[1];

        _medoids[0] = (uint32_t)(medoid_id_on_file);

    } else {

        LOG_INFO("medoids file exists, loading medoids ...");

        size_t tmp_dim;

        load_bin<uint32_t>(medoids_data_path, _medoids, _num_medoids, tmp_dim);

        if (tmp_dim != 1)
        {
            LOG_ERROR("Error loading medoids file. Expected bin format of m times 1 vector of uint32_t.");
            return -1;
        }
    }

    use_medoids_data_as_centroids();

    return 0;
}

void DiskVamanaIndex::cached_beam_search(const void *query, const uint64_t k_search, const uint64_t l_search,
                                                 uint64_t *indices, float *distances, const uint64_t beam_width,
                                                 QueryStats *stats, const uint32_t io_limit) {
    uint64_t num_sector_per_nodes = DIV_ROUND_UP(_max_node_len, defaults::SECTOR_LEN);
    if (beam_width > num_sector_per_nodes * defaults::MAX_N_SECTOR_READS) {
        throw std::runtime_error("Beamwidth can not be higher than defaults::MAX_N_SECTOR_READS");
    }

    ScratchStoreManager<SSDThreadData> manager(this->_thread_data);
    auto data = manager.scratch_space();
    IOContext &ctx = data->ctx;
    auto query_scratch = &(data->scratch);
    auto pq_query_scratch = query_scratch->_pq_scratch;

    // reset query scratch
    query_scratch->reset();

    // copy query to thread specific aligned and allocated memory (for distance
    // calculations we need aligned data)
    void *aligned_query_T = query_scratch->aligned_query_T;
    
    for (size_t i = 0; i < this->data_dim_; i++)
    {
        memcpy((char *)aligned_query_T + i * data_size_, (char *)query + i * data_size_, data_size_);
    }

    // pointers to buffers for data
    float *data_buf = (float *)query_scratch->coord_scratch;
    half_float::half *data_buf_half = (half_float::half *)query_scratch->coord_scratch;

    // sector scratch
    char *sector_scratch = query_scratch->sector_scratch;
    uint64_t &sector_scratch_idx = query_scratch->sector_idx;
    const uint64_t num_sectors_per_node =
        _nnodes_per_sector > 0 ? 1 : DIV_ROUND_UP(_max_node_len, defaults::SECTOR_LEN);

    // query <-> neighbor list
    float *dist_scratch = pq_query_scratch->aligned_dist_scratch;
    uint16_t *pq_coord_scratch = pq_query_scratch->aligned_pq_coord_scratch;

    Timer query_timer, io_timer, cpu_timer;

    tsl::robin_set<uint64_t> &visited = query_scratch->visited;
    NeighborPriorityQueue &retset = query_scratch->retset;
    retset.reserve(l_search);
    std::vector<Neighbor> &full_retset = query_scratch->full_retset;

    uint32_t best_medoid = 0;
    float best_dist = (std::numeric_limits<float>::max)();

    for (uint64_t cur_m = 0; cur_m < _num_medoids; cur_m++)
    {
        float cur_expanded_dist = this->compare((const void *)aligned_query_T, (const void *)(_centroid_data + aligned_dim_ * cur_m), (uint32_t)aligned_dim_);

        if (cur_expanded_dist < best_dist)
        {
            best_medoid = _medoids[cur_m];
            best_dist = cur_expanded_dist;
        }
    }

    compute_pq_dists((const void *)query, &best_medoid, 1, dist_scratch);

    retset.insert(Neighbor(best_medoid, dist_scratch[0]));
    visited.insert(best_medoid);

    uint32_t cmps = 0;
    uint32_t hops = 0;
    uint32_t num_ios = 0;

    // cleared every iteration
    std::vector<uint32_t> frontier;
    frontier.reserve(2 * beam_width);
    std::vector<std::pair<uint32_t, char *>> frontier_nhoods;
    frontier_nhoods.reserve(2 * beam_width);
    std::vector<AlignedRead> frontier_read_reqs;
    frontier_read_reqs.reserve(2 * beam_width);
    std::vector<std::pair<uint32_t, std::pair<uint32_t, uint32_t *>>> cached_nhoods;
    cached_nhoods.reserve(2 * beam_width);

    while (retset.has_unexpanded_node() && num_ios < io_limit)
    {
        // clear iteration state
        frontier.clear();
        frontier_nhoods.clear();
        frontier_read_reqs.clear();
        cached_nhoods.clear();
        sector_scratch_idx = 0;
        // find new beam
        uint32_t num_seen = 0;
        while (retset.has_unexpanded_node() && frontier.size() < beam_width && num_seen < beam_width)
        {
            auto nbr = retset.closest_unexpanded();
            num_seen++;
            auto iter = nhood_cache_->find(nbr.id);
            if (iter != nhood_cache_->end())
            {
                cached_nhoods.push_back(std::make_pair(nbr.id, iter->second));
                if (stats != nullptr)
                {
                    stats->n_cache_hits++;
                }
            }
            else
            {
                frontier.push_back(nbr.id);
            }
        }

        // read nhoods of frontier ids
        if (!frontier.empty())
        {
            if (stats != nullptr)
                stats->n_hops++;
            for (uint64_t i = 0; i < frontier.size(); i++)
            {
                auto id = frontier[i];
                std::pair<uint32_t, char *> fnhood;
                fnhood.first = id;
                fnhood.second = sector_scratch + num_sectors_per_node * sector_scratch_idx * defaults::SECTOR_LEN;
                sector_scratch_idx++;
                frontier_nhoods.push_back(fnhood);
                frontier_read_reqs.emplace_back(get_node_sector((size_t)id) * defaults::SECTOR_LEN,
                                                num_sectors_per_node * defaults::SECTOR_LEN, fnhood.second);
                if (stats != nullptr)
                {
                    stats->n_4k++;
                    stats->n_ios++;
                }
                num_ios++;
            }
            io_timer.reset();
            reader->read(frontier_read_reqs, ctx); // synchronous IO linux
            if (stats != nullptr)
            {
                stats->io_us += (float)io_timer.elapsed();
            }
        }

        // process cached nhoods
        for (auto &cached_nhood : cached_nhoods)
        {
            float cur_expanded_dist;
            if (_use_half) {
                auto global_cache_iter = coord_cache_half_->find(cached_nhood.first);
                half_float::half *node_hf_coords_copy = global_cache_iter->second;
                cur_expanded_dist = this->compare((const void *)aligned_query_T, (const void *)node_hf_coords_copy, (uint32_t)aligned_dim_);
            } else {
                auto global_cache_iter = coord_cache_->find(cached_nhood.first);
                float *node_fp_coords_copy = global_cache_iter->second;
                cur_expanded_dist = this->compare((const void *)aligned_query_T, (const void *)node_fp_coords_copy, (uint32_t)aligned_dim_);
            }

            full_retset.push_back(Neighbor((uint32_t)cached_nhood.first, cur_expanded_dist));

            uint64_t nnbrs = cached_nhood.second.first;
            uint32_t *node_nbrs = cached_nhood.second.second;

            cpu_timer.reset();
            compute_pq_dists((const void *)query, node_nbrs, nnbrs, dist_scratch);

            if (stats != nullptr)
            {
                stats->n_cmps += (uint32_t)nnbrs;
                stats->cpu_us += (float)cpu_timer.elapsed();
            }

            // process prefetched nhood
            for (uint64_t m = 0; m < nnbrs; ++m)
            {
                uint32_t id = node_nbrs[m];
                if (visited.insert(id).second)
                {
                    cmps++;
                    float dist = dist_scratch[m];
                    Neighbor nn(id, dist);
                    retset.insert(nn);
                }
            }
        }

        for (auto &frontier_nhood : frontier_nhoods)
        {
            char *node_disk_buf = offset_to_node(frontier_nhood.second, frontier_nhood.first);
            uint32_t *node_buf = offset_to_node_nhood(node_disk_buf);
            uint64_t nnbrs = (uint64_t)(*node_buf);
            float *node_fp_coords;
            half_float::half *node_hf_coords;
            float cur_expanded_dist;
            if (_use_half) {
                node_hf_coords = offset_to_node_coords_half(node_disk_buf);
                memcpy(data_buf_half, node_hf_coords, _disk_bytes_per_point);
                cur_expanded_dist = this->compare((const void *)aligned_query_T, (const void *)data_buf_half, (uint32_t)aligned_dim_);
            } else {
                node_fp_coords = offset_to_node_coords(node_disk_buf);
                memcpy(data_buf, node_fp_coords, _disk_bytes_per_point);
                cur_expanded_dist = this->compare((const void *)aligned_query_T, (const void *)data_buf, (uint32_t)aligned_dim_);
            }

            full_retset.push_back(Neighbor(frontier_nhood.first, cur_expanded_dist));
            uint32_t *node_nbrs = (node_buf + 1);
            cpu_timer.reset();
            compute_pq_dists((const void *)query, node_nbrs, nnbrs, dist_scratch);
            if (stats != nullptr)
            {
                stats->n_cmps += (uint32_t)nnbrs;
                stats->cpu_us += (float)cpu_timer.elapsed();
            }

            cpu_timer.reset();
            // process prefetch-ed nhood
            for (uint64_t m = 0; m < nnbrs; ++m)
            {
                uint32_t id = node_nbrs[m];
                if (visited.insert(id).second)
                {
                    cmps++;
                    float dist = dist_scratch[m];
                    if (stats != nullptr)
                    {
                        stats->n_cmps++;
                    }

                    Neighbor nn(id, dist);
                    retset.insert(nn);
                }
            }

            if (stats != nullptr)
            {
                stats->cpu_us += (float)cpu_timer.elapsed();
            }
        }

        hops++;
    }

    // re-sort by distance
    std::sort(full_retset.begin(), full_retset.end());

    // copy k_search values
    for (uint64_t i = 0; i < k_search; i++)
    {
        indices[i] = full_retset[i].id;
        auto key = (uint32_t)indices[i];

        if (distances != nullptr)
        {
            distances[i] = full_retset[i].distance;
        }
    }


    if (stats != nullptr)
    {
        stats->total_us = (float)query_timer.elapsed();
        stats->n_hops = hops;
        stats->n_cmps = cmps;
        stats->n_ios = num_ios;
    }
}

void DiskVamanaIndex::cached_beam_search_half(const void *query, const void *query_raw, const uint64_t k_search, const uint64_t l_search,
                                                 uint64_t *indices, float *distances, const uint64_t beam_width,
                                                 QueryStats *stats, const uint32_t io_limit) {
    uint64_t num_sector_per_nodes = DIV_ROUND_UP(_max_node_len, defaults::SECTOR_LEN);
    if (beam_width > num_sector_per_nodes * defaults::MAX_N_SECTOR_READS) {
        throw std::runtime_error("Beamwidth can not be higher than defaults::MAX_N_SECTOR_READS");
    }

    ScratchStoreManager<SSDThreadData> manager(this->_thread_data);
    auto data = manager.scratch_space();
    IOContext &ctx = data->ctx;
    auto query_scratch = &(data->scratch);
    auto pq_query_scratch = query_scratch->_pq_scratch;

    // reset query scratch
    query_scratch->reset();

    // copy query to thread specific aligned and allocated memory (for distance
    // calculations we need aligned data)
    void *aligned_query_T = query_scratch->aligned_query_T;

    for (size_t i = 0; i < this->data_dim_; i++)
    {
        memcpy((char *)aligned_query_T + i * data_size_, (char *)query + i * data_size_, data_size_);
    }

    // pointers to buffers for data
    float *data_buf = (float *)query_scratch->coord_scratch;
    half_float::half *data_buf_half = (half_float::half *)query_scratch->coord_scratch;

    // sector scratch
    char *sector_scratch = query_scratch->sector_scratch;
    uint64_t &sector_scratch_idx = query_scratch->sector_idx;
    const uint64_t num_sectors_per_node =
        _nnodes_per_sector > 0 ? 1 : DIV_ROUND_UP(_max_node_len, defaults::SECTOR_LEN);

    // query <-> neighbor list
    float *dist_scratch = pq_query_scratch->aligned_dist_scratch;
    uint16_t *pq_coord_scratch = pq_query_scratch->aligned_pq_coord_scratch;

    Timer query_timer, io_timer, cpu_timer;

    tsl::robin_set<uint64_t> &visited = query_scratch->visited;
    NeighborPriorityQueue &retset = query_scratch->retset;
    retset.reserve(l_search);
    std::vector<Neighbor> &full_retset = query_scratch->full_retset;

    uint32_t best_medoid = 0;
    float best_dist = (std::numeric_limits<float>::max)();

    for (uint64_t cur_m = 0; cur_m < _num_medoids; cur_m++)
    {
        float cur_expanded_dist = this->compare((const void *)aligned_query_T, (const void *)(_centroid_half_data + aligned_dim_ * cur_m), (uint32_t)aligned_dim_); 

        if (cur_expanded_dist < best_dist)
        {
            best_medoid = _medoids[cur_m];
            best_dist = cur_expanded_dist;
        }
    }

    compute_pq_dists((const void *)query_raw, &best_medoid, 1, dist_scratch);

    retset.insert(Neighbor(best_medoid, dist_scratch[0]));
    visited.insert(best_medoid);

    uint32_t cmps = 0;
    uint32_t hops = 0;
    uint32_t num_ios = 0;

    // cleared every iteration
    std::vector<uint32_t> frontier;
    frontier.reserve(2 * beam_width);
    std::vector<std::pair<uint32_t, char *>> frontier_nhoods;
    frontier_nhoods.reserve(2 * beam_width);
    std::vector<AlignedRead> frontier_read_reqs;
    frontier_read_reqs.reserve(2 * beam_width);
    std::vector<std::pair<uint32_t, std::pair<uint32_t, uint32_t *>>> cached_nhoods;
    cached_nhoods.reserve(2 * beam_width);

    while (retset.has_unexpanded_node() && num_ios < io_limit)
    {
        // clear iteration state
        frontier.clear();
        frontier_nhoods.clear();
        frontier_read_reqs.clear();
        cached_nhoods.clear();
        sector_scratch_idx = 0;
        // find new beam
        uint32_t num_seen = 0;
        while (retset.has_unexpanded_node() && frontier.size() < beam_width && num_seen < beam_width)
        {
            auto nbr = retset.closest_unexpanded();
            num_seen++;
            auto iter = nhood_cache_->find(nbr.id);
            if (iter != nhood_cache_->end())
            {
                cached_nhoods.push_back(std::make_pair(nbr.id, iter->second));
                if (stats != nullptr)
                {
                    stats->n_cache_hits++;
                }
            }
            else
            {
                frontier.push_back(nbr.id);
            }
        }

        // read nhoods of frontier ids
        if (!frontier.empty())
        {
            if (stats != nullptr)
                stats->n_hops++;
            for (uint64_t i = 0; i < frontier.size(); i++)
            {
                auto id = frontier[i];
                std::pair<uint32_t, char *> fnhood;
                fnhood.first = id;
                fnhood.second = sector_scratch + num_sectors_per_node * sector_scratch_idx * defaults::SECTOR_LEN;
                sector_scratch_idx++;
                frontier_nhoods.push_back(fnhood);
                frontier_read_reqs.emplace_back(get_node_sector((size_t)id) * defaults::SECTOR_LEN,
                                                num_sectors_per_node * defaults::SECTOR_LEN, fnhood.second);
                if (stats != nullptr)
                {
                    stats->n_4k++;
                    stats->n_ios++;
                }
                num_ios++;
            }
            io_timer.reset();
            reader->read(frontier_read_reqs, ctx); // synchronous IO linux
            if (stats != nullptr)
            {
                stats->io_us += (float)io_timer.elapsed();
            }
        }

        // process cached nhoods
        for (auto &cached_nhood : cached_nhoods)
        {
            float cur_expanded_dist;
            if (_use_half) {
                auto global_cache_iter = coord_cache_half_->find(cached_nhood.first);
                half_float::half *node_hf_coords_copy = global_cache_iter->second;
                cur_expanded_dist = this->compare((const void *)aligned_query_T, (const void *)node_hf_coords_copy, (uint32_t)aligned_dim_);
            } else {
                auto global_cache_iter = coord_cache_->find(cached_nhood.first);
                float *node_fp_coords_copy = global_cache_iter->second;
                cur_expanded_dist = this->compare((const void *)aligned_query_T, (const void *)node_fp_coords_copy, (uint32_t)aligned_dim_);
            }

            full_retset.push_back(Neighbor((uint32_t)cached_nhood.first, cur_expanded_dist));

            uint64_t nnbrs = cached_nhood.second.first;
            uint32_t *node_nbrs = cached_nhood.second.second;

            cpu_timer.reset();
            compute_pq_dists((const void *)query_raw, node_nbrs, nnbrs, dist_scratch);

            if (stats != nullptr)
            {
                stats->n_cmps += (uint32_t)nnbrs;
                stats->cpu_us += (float)cpu_timer.elapsed();
            }

            // process prefetched nhood
            for (uint64_t m = 0; m < nnbrs; ++m)
            {
                uint32_t id = node_nbrs[m];
                if (visited.insert(id).second)
                {
                    cmps++;
                    float dist = dist_scratch[m];
                    Neighbor nn(id, dist);
                    retset.insert(nn);
                }
            }
        }

        for (auto &frontier_nhood : frontier_nhoods)
        {
            char *node_disk_buf = offset_to_node(frontier_nhood.second, frontier_nhood.first);
            uint32_t *node_buf = offset_to_node_nhood(node_disk_buf);
            uint64_t nnbrs = (uint64_t)(*node_buf);
            float *node_fp_coords;
            half_float::half *node_hf_coords;
            float cur_expanded_dist;
            if (_use_half) {
                node_hf_coords = offset_to_node_coords_half(node_disk_buf);
                memcpy(data_buf_half, node_hf_coords, _disk_bytes_per_point);
                cur_expanded_dist = this->compare((const void *)aligned_query_T, (const void *)data_buf_half, (uint32_t)aligned_dim_);
            } else {
                node_fp_coords = offset_to_node_coords(node_disk_buf);
                memcpy(data_buf, node_fp_coords, _disk_bytes_per_point);
                cur_expanded_dist = this->compare((const void *)aligned_query_T, (const void *)data_buf, (uint32_t)aligned_dim_);
            }

            full_retset.push_back(Neighbor(frontier_nhood.first, cur_expanded_dist));
            uint32_t *node_nbrs = (node_buf + 1);
            cpu_timer.reset();
            compute_pq_dists((const void *)query_raw, node_nbrs, nnbrs, dist_scratch);
            if (stats != nullptr)
            {
                stats->n_cmps += (uint32_t)nnbrs;
                stats->cpu_us += (float)cpu_timer.elapsed();
            }

            cpu_timer.reset();
            // process prefetch-ed nhood
            for (uint64_t m = 0; m < nnbrs; ++m)
            {
                uint32_t id = node_nbrs[m];
                if (visited.insert(id).second)
                {
                    cmps++;
                    float dist = dist_scratch[m];
                    if (stats != nullptr)
                    {
                        stats->n_cmps++;
                    }

                    Neighbor nn(id, dist);
                    retset.insert(nn);
                }
            }

            if (stats != nullptr)
            {
                stats->cpu_us += (float)cpu_timer.elapsed();
            }
        }

        hops++;
    }

    // re-sort by distance
    std::sort(full_retset.begin(), full_retset.end());

    // copy k_search values
    for (uint64_t i = 0; i < k_search; i++)
    {
        indices[i] = full_retset[i].id;
        if (distances != nullptr)
        {
            distances[i] = full_retset[i].distance;
        }
    }

    if (stats != nullptr)
    {
        stats->total_us = (float)query_timer.elapsed();
        stats->n_hops = hops;
        stats->n_cmps = cmps;
        stats->n_ios = num_ios;
    }
}

void DiskVamanaIndex::use_medoids_data_as_centroids() {
    if (_use_half) {
        if (_centroid_half_data != nullptr)
            aligned_free(_centroid_half_data);
        alloc_aligned(((void **)&_centroid_half_data), _num_medoids * aligned_dim_ * data_size_, 32);
        std::memset(_centroid_half_data, 0, _num_medoids * aligned_dim_ * data_size_);
    } else {
        if (_centroid_data != nullptr)
            aligned_free(_centroid_data);
        alloc_aligned(((void **)&_centroid_data), _num_medoids * aligned_dim_ * data_size_, 32);
        std::memset(_centroid_data, 0, _num_medoids * aligned_dim_ * data_size_);
    }

    LOG_INFO("Loading centroid data from medoids vector data of %lu medoid(s): ", _num_medoids);

    for (uint64_t i = 0; i < _num_medoids; i++)
    {
        LOG_INFO("medoids[%lu] = %d", i, _medoids[i]);
    }

    // borrow ctx
    ScratchStoreManager<SSDThreadData> manager(this->_thread_data);
    auto data = manager.scratch_space();
    IOContext &ctx = data->ctx;

    std::vector<uint32_t> nodes_to_read;
    std::vector<float *> medoid_bufs;
    std::vector<half_float::half *> medoid_bufs_half;
    std::vector<std::pair<uint32_t, uint32_t *>> nbr_bufs;

    for (uint64_t cur_m = 0; cur_m < _num_medoids; cur_m++)
    {
        nodes_to_read.push_back(_medoids[cur_m]);
        if (_use_half) {
            medoid_bufs_half.push_back(new half_float::half[data_dim_]);
        } else {
            medoid_bufs.push_back(new float[data_dim_]);
        }
        nbr_bufs.emplace_back(0, nullptr);
    }

    auto read_status = read_nodes(nodes_to_read, medoid_bufs, medoid_bufs_half, nbr_bufs);

    for (uint64_t cur_m = 0; cur_m < _num_medoids; cur_m++)
    {
        if (read_status[cur_m] == true)
        {
            if (_use_half) {
                for (uint32_t i = 0; i < data_dim_; i++)
                    _centroid_half_data[cur_m * aligned_dim_ + i] = medoid_bufs_half[cur_m][i];
            } else {
                for (uint32_t i = 0; i < data_dim_; i++)
                    _centroid_data[cur_m * aligned_dim_ + i] = medoid_bufs[cur_m][i];
            }
        }
        else
        {
            throw std::runtime_error("Unable to read a medoid!!!");
        }
        if (_use_half)
        {
            delete[] medoid_bufs_half[cur_m];
        }
        else
        {
            delete[] medoid_bufs[cur_m];
        }
    }
}

std::vector<bool> DiskVamanaIndex::read_nodes(const std::vector<uint32_t> &node_ids,
                                    std::vector<float *> &coord_buffers,
                                    std::vector<half_float::half *> &coord_buffers_half,
                                    std::vector<std::pair<uint32_t, uint32_t *>> &nbr_buffers) {
    std::vector<AlignedRead> read_reqs;
    std::vector<bool> retval(node_ids.size(), true);

    char *buf = nullptr;
    auto num_sectors = _nnodes_per_sector > 0 ? 1 : DIV_ROUND_UP(_max_node_len, defaults::SECTOR_LEN);
    alloc_aligned((void **)&buf, node_ids.size() * num_sectors * defaults::SECTOR_LEN, defaults::SECTOR_LEN);

    // create read requests
    for (size_t i = 0; i < node_ids.size(); ++i)
    {
        auto node_id = node_ids[i];

        AlignedRead read;
        read.len = num_sectors * defaults::SECTOR_LEN;
        read.buf = buf + i * num_sectors * defaults::SECTOR_LEN;
        read.offset = get_node_sector(node_id) * defaults::SECTOR_LEN;
        read_reqs.push_back(read);
    }

    // // borrow thread data and issue reads
    ScratchStoreManager<SSDThreadData> manager(this->_thread_data);
    auto this_thread_data = manager.scratch_space();
    IOContext &ctx = this_thread_data->ctx;
    reader->read(read_reqs, ctx);

    // copy reads into buffers
    for (uint32_t i = 0; i < read_reqs.size(); i++)
    {
        char *node_buf = offset_to_node((char *)read_reqs[i].buf, node_ids[i]);

        if (_use_half) {
            if (coord_buffers_half[i] != nullptr)
            {
                half_float::half *node_coords = offset_to_node_coords_half(node_buf);
                memcpy(coord_buffers_half[i], node_coords, _disk_bytes_per_point);
            }
        } else {
            if (coord_buffers[i] != nullptr)
            {
                float *node_coords = offset_to_node_coords(node_buf);
                memcpy(coord_buffers[i], node_coords, _disk_bytes_per_point);
            }
        }

        if (nbr_buffers[i].second != nullptr)
        {
            uint32_t *node_nhood = offset_to_node_nhood(node_buf);
            auto num_nbrs = *node_nhood;
            nbr_buffers[i].first = num_nbrs;
            memcpy(nbr_buffers[i].second, node_nhood + 1, num_nbrs * sizeof(uint32_t));
        }
    }

    aligned_free(buf);

    return retval;
}

void DiskVamanaIndex::compute_pq_dists(const void *query, const uint32_t *ids, const uint64_t n_ids, float *dists_out)
{
    QueryDistanceMatrix qdm(index_meta_L2_, &GetPqCentroidResource());
    qdm.initDistanceMatrix(query);
    PqDistScorer scorer(&this->pq_code_profile_);
    for (size_t i = 0; i < n_ids; i++)
    {
        dists_out[i] = scorer.score(ids[i], &qdm);
    }
}

MatrixPointer DiskVamanaIndex::NullMatrix() const {
    return MatrixPointer(nullptr);
}

int16_t DiskVamanaIndex::GetOrDefault(const IndexParams& params, const std::string& key, const uint16_t default_value) {
    if (params.has(key)) {
        return params.getUint16(key);
    }

    return default_value;
}

bool DiskVamanaIndex::StrToValue(const std::string& source, void* value) const {
    switch (index_meta_L2_.type()) {
    case IndexMeta::FeatureTypes::kTypeUnknown:
        return false;
    case IndexMeta::FeatureTypes::kTypeBinary:
        //TODO, how to deal with binary
        return false;
    case IndexMeta::FeatureTypes::kTypeHalfFloat:
        if (!StringUtil::strToHalf(source.c_str(), *(half_float::half*)value)) {
            return false;
        }
        return true;
    case IndexMeta::FeatureTypes::kTypeFloat:
        if (!StringUtil::strToFloat(source.c_str(), *(float*)value)) {
            return false;
        }
        return true;
    case IndexMeta::FeatureTypes::kTypeDouble:
        if (!StringUtil::strToDouble(source.c_str(), *(double*)value)) {
            return false;
        }
        return true;
    case IndexMeta::FeatureTypes::kTypeInt8:
        if (!StringUtil::strToInt8(source.c_str(), *(int8_t*)value)) {
            return false;
        }
        return true;
    case IndexMeta::FeatureTypes::kTypeInt16:
        if (!StringUtil::strToInt16(source.c_str(), *(int16_t*)value)) {
            return false;
        }
        return true;
    }

    return false;
}

char * DiskVamanaIndex::offset_to_node(char *sector_buf, uint64_t node_id) {
    return sector_buf + (_nnodes_per_sector == 0 ? 0 : (node_id % _nnodes_per_sector) * _max_node_len);
}

uint32_t * DiskVamanaIndex::offset_to_node_nhood(char *node_buf) {
    return (unsigned *)(node_buf + _disk_bytes_per_point);
}

float * DiskVamanaIndex::offset_to_node_coords(char *node_buf) {
    return (float *)(node_buf);
}

half_float::half * DiskVamanaIndex::offset_to_node_coords_half(char *node_buf) {
    return (half_float::half *)(node_buf);
}

uint64_t DiskVamanaIndex::get_node_sector(uint64_t node_id) {
    return 1 + (_nnodes_per_sector > 0 ? node_id / _nnodes_per_sector
                                       : node_id * DIV_ROUND_UP(_max_node_len, defaults::SECTOR_LEN));
}

void DiskVamanaIndex::GetBaseVec(docid_t doc_id, void *dest)
{
    coarseVamanaIndex_->get_data_vec(doc_id, dest);
}

size_t DiskVamanaIndex::GetBaseVecNum()
{
    return coarseVamanaIndex_->get_active_num();
}

MERCURY_NAMESPACE_END(core);