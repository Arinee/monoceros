#pragma once

#include <sstream>
#include <sys/stat.h>
#include "src/core/common/common.h"
#include "cached_io.h"
#include "src/core/framework/index_framework.h"

MERCURY_NAMESPACE_BEGIN(core);

#define IS_ALIGNED(X, Y) ((uint64_t)(X) % (uint64_t)(Y) == 0)

#define ROUND_UP(X, Y) ((((uint64_t)(X) / (Y)) + ((uint64_t)(X) % (Y) != 0)) * (Y))

// setting it to 8 because that works well for AVX2. If we have AVX512
// implementations of distance algos, they might have to set this to 16

inline void print_error_and_terminate(std::stringstream &error_stream)
{
    std::cerr << error_stream.str() << std::endl;
    throw new std::runtime_error(error_stream.str());
}

inline size_t get_file_size(const std::string &fname)
{
    std::ifstream reader(fname, std::ios::binary | std::ios::ate);
    if (!reader.fail() && reader.is_open())
    {
        size_t end_pos = reader.tellg();
        reader.close();
        return end_pos;
    }
    else
    {
        std::cerr << "Could not open file: " << fname << std::endl;
        return 0;
    }
}

inline void report_memory_allocation_failure()
{
    std::stringstream stream;
    stream << "Memory Allocation Failed.";
    print_error_and_terminate(stream);
}

inline void report_misalignment_of_requested_size(size_t align)
{
    std::stringstream stream;
    stream << "Requested memory size is not a multiple of " << align << ". Can not be allocated.";
    print_error_and_terminate(stream);
}

inline void alloc_aligned(void **ptr, size_t size, size_t align)
{
    *ptr = nullptr;
    if (IS_ALIGNED(size, align) == 0)
        report_misalignment_of_requested_size(align);
    *ptr = ::aligned_alloc(align, size);
    if (*ptr == nullptr)
        report_memory_allocation_failure();
}

inline void aligned_free(void *ptr)
{
    if (ptr == nullptr)
    {
        return;
    }
    free(ptr);
}

inline bool file_exists(const std::string &name, bool dirCheck = false)
{
    int val;
    struct stat buffer;
    val = stat(name.c_str(), &buffer);

    if (val != 0)
    {
        switch (errno)
        {
        case EINVAL:
            LOG_ERROR("Invalid argument passed to stat()");
            break;
        case ENOENT:
            // file is not existing, not an issue, so we won't cout anything.
            break;
        default:
            LOG_ERROR("Unexpected error in stat():");
            break;
        }
        return false;
    }
    else
    {
        // the file entry exists. If reqd, check if this is a directory.
        return dirCheck ? buffer.st_mode & S_IFDIR : true;
    }
}

inline void process_mem_usage(double& vm_usage, double& resident_set)
{
    vm_usage     = 0.0;
    resident_set = 0.0;

    // the two fields we want
    unsigned long vsize;
    long rss;
    {
        std::string ignore;
        std::ifstream ifs("/proc/self/stat", std::ios_base::in);
        ifs >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore
                >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore
                >> ignore >> ignore >> vsize >> rss;
    }

    long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // in case x86-64 is configured to use 2MB pages
    vm_usage = vsize / 1024.0;
    resident_set = rss * page_size_kb;
}

inline int delete_file(const std::string &fileName)
{
    if (file_exists(fileName))
    {
        auto rc = ::remove(fileName.c_str());
        if (rc != 0)
        {
            std::cerr << "Could not delete file: " << fileName
                          << " even though it exists. This might indicate a permissions "
                             "issue. "
                             "If you see this message, please contact the diskann team."
                          << std::endl;
        }
        return rc;
    }
    else
    {
        return 0;
    }
}

inline void open_file_to_write(std::ofstream &writer, const std::string &filename)
{
    writer.exceptions(std::ofstream::failbit | std::ofstream::badbit);
    if (!file_exists(filename))
        writer.open(filename, std::ios::binary | std::ios::out);
    else
        writer.open(filename, std::ios::binary | std::ios::in | std::ios::out);

    if (writer.fail())
    {
        char buff[1024];
        auto ret = std::string(strerror_r(errno, buff, 1024));
        std::string error_message =
            std::string("Failed to open file") + filename + " for write because " + buff + ", ret=" + ret;
        std::cerr << error_message << std::endl;
        throw new std::runtime_error(error_message);
    }
}

inline size_t save_data_in_base_dimensions(uint16_t data_size, const std::string &filename, void *data, size_t npts, size_t ndims,
                                           size_t aligned_dim, size_t offset = 0)
{
    std::ofstream writer; //(filename, std::ios::binary | std::ios::out);
    open_file_to_write(writer, filename);
    int npts_i32 = (int)npts, ndims_i32 = (int)ndims;
    size_t bytes_written = 2 * sizeof(uint32_t) + npts * ndims * data_size;
    writer.seekp(offset, writer.beg);
    writer.write((char *)&npts_i32, sizeof(int));
    writer.write((char *)&ndims_i32, sizeof(int));
    for (size_t i = 0; i < npts; i++)
    {
        writer.write(((char *)data + i * aligned_dim * data_size), ndims * data_size);
    }
    writer.close();
    return bytes_written;
}

template <typename T>
inline void load_bin_impl(std::basic_istream<char> &reader, T *&data, size_t &npts, size_t &dim, size_t file_offset = 0)
{
    int npts_i32, dim_i32;

    reader.seekg(file_offset, reader.beg);
    reader.read((char *)&npts_i32, sizeof(int));
    reader.read((char *)&dim_i32, sizeof(int));
    npts = (unsigned)npts_i32;
    dim = (unsigned)dim_i32;

    std::cout << "Metadata: #pts = " << npts << ", #dims = " << dim << "..." << std::endl;

    data = new T[npts * dim];
    reader.read((char *)data, npts * dim * sizeof(T));
}

template <typename T>
inline void load_bin(const std::string &bin_file, T *&data, size_t &npts, size_t &dim, size_t offset = 0)
{
    std::cout << "Reading bin file " << bin_file.c_str() << " ..." << std::endl;
    std::ifstream reader;
    reader.exceptions(std::ifstream::failbit | std::ifstream::badbit);

    try
    {
        std::cout << "Opening bin file " << bin_file.c_str() << "... " << std::endl;
        reader.open(bin_file, std::ios::binary | std::ios::ate);
        reader.seekg(0);
        load_bin_impl<T>(reader, data, npts, dim, offset);
    }
    catch (std::system_error &e)
    {
        throw std::io_errc();
    }
    std::cout << "done." << std::endl;
}

inline void read_idmap(const std::string &fname, std::vector<uint32_t> &ivecs)
{
    uint32_t npts32, dim;
    size_t actual_file_size = get_file_size(fname);
    std::ifstream reader(fname.c_str(), std::ios::binary);
    reader.read((char *)&npts32, sizeof(uint32_t));
    reader.read((char *)&dim, sizeof(uint32_t));
    if (dim != 1 || actual_file_size != ((size_t)npts32) * sizeof(uint32_t) + 2 * sizeof(uint32_t))
    {
        std::stringstream stream;
        stream << "Error reading idmap file. Check if the file is bin file with "
                  "1 dimensional data. Actual: "
               << actual_file_size << ", expected: " << (size_t)npts32 + 2 * sizeof(uint32_t) << std::endl;

        throw std::runtime_error(stream.str());
    }
    ivecs.resize(npts32);
    reader.read((char *)ivecs.data(), ((size_t)npts32) * sizeof(uint32_t));
    reader.close();
}

inline void get_bin_metadata_impl(std::basic_istream<char> &reader, size_t &nrows, size_t &ncols, size_t offset = 0)
{
    int nrows_32, ncols_32;
    reader.seekg(offset, reader.beg);
    reader.read((char *)&nrows_32, sizeof(int));
    reader.read((char *)&ncols_32, sizeof(int));
    nrows = nrows_32;
    ncols = ncols_32;
}

inline void get_bin_metadata(const std::string &bin_file, size_t &nrows, size_t &ncols, size_t offset = 0)
{
    std::ifstream reader(bin_file.c_str(), std::ios::binary);
    get_bin_metadata_impl(reader, nrows, ncols, offset);
}

inline void copy_aligned_data_from_file(uint16_t data_size, const char *bin_file, void *&data, size_t &npts, size_t &dim,
                                        const size_t &rounded_dim, size_t offset = 0)
{
    if (data == nullptr)
    {
        LOG_ERROR("Null pointer passed to copy_aligned_data_from_file function");
        std::stringstream stream;
        stream << "Memory was not allocated for " << data << " before calling the load function. Exiting..." << std::endl;;
        throw new std::runtime_error(stream.str());
    }
    std::ifstream reader;
    reader.exceptions(std::ios::badbit | std::ios::failbit);
    reader.open(bin_file, std::ios::binary);
    reader.seekg(offset, reader.beg);

    int npts_i32, dim_i32;
    reader.read((char *)&npts_i32, sizeof(int));
    reader.read((char *)&dim_i32, sizeof(int));
    npts = (unsigned)npts_i32;
    dim = (unsigned)dim_i32;

    for (size_t i = 0; i < npts; i++)
    {
        reader.read(((char *)data + i * rounded_dim * data_size), dim * data_size);
        memset((char *)data + i * rounded_dim * data_size + dim, 0, (rounded_dim - dim) * data_size);
    }
}

inline void open_text_file_to_write(std::ofstream &writer, const std::string &filename)
{
    writer.exceptions(std::ofstream::failbit | std::ofstream::badbit);
    if (!file_exists(filename))
        writer.open(filename, std::ios::out);
    else
        writer.open(filename, std::ios::in | std::ios::out);

    if (writer.fail())
    {
        char buff[1024];
        auto ret = std::string(strerror_r(errno, buff, 1024));
        std::string error_message =
            std::string("Failed to open file") + filename + " for write because " + buff + ", ret=" + ret;
        std::cerr << error_message << std::endl;
        throw new std::runtime_error(error_message);
    }
}

template <typename T>
inline void generate_data_for_PQ_train(std::string &out_file, std::string &bin_file, size_t &npts, size_t &dim, size_t offset = 0)
{
    std::ifstream reader;
    reader.exceptions(std::ios::badbit | std::ios::failbit);
    reader.open(bin_file, std::ios::binary);
    reader.seekg(offset, reader.beg);

    int npts_i32, dim_i32;
    reader.read((char *)&npts_i32, sizeof(int));
    reader.read((char *)&dim_i32, sizeof(int));
    npts = (unsigned)npts_i32;
    dim = (unsigned)dim_i32;

    T data[dim];

    std::ofstream writer;

    open_text_file_to_write(writer, out_file);

    for (size_t i = 0; i < npts; i++)
    {
        reader.read(((char *)data), dim * sizeof(T));
        writer << "CMD=add" << char(31) << "" << char(10);
        writer << "id=" << i << char(31) << "" << char(10);
        writer << "cate_vec=";
        for (size_t j = 0; j < dim; j++) {
            if (j != dim - 1) {
                writer << data[j] << " ";
            } else {
                writer << data[j] << char(31) << "" << char(10);
            }
        }
        writer << char(30) << char(10);
    }
    writer.close();
}

// NOTE :: good efficiency when total_vec_size is integral multiple of 64
inline void prefetch_vector_impl(const char *vec, size_t vecsize)
{
    size_t max_prefetch_size = (vecsize / 64) * 64;
    for (size_t d = 0; d < max_prefetch_size; d += 64)
        _mm_prefetch((const char *)vec + d, _MM_HINT_T0);
}

inline void load_aligned_bin_impl(uint16_t data_size, std::basic_istream<char> &reader, size_t actual_file_size, void *&data, size_t &npts,
                                  size_t &dim, size_t &rounded_dim)
{
    int npts_i32, dim_i32;
    reader.read((char *)&npts_i32, sizeof(int));
    reader.read((char *)&dim_i32, sizeof(int));
    npts = (unsigned)npts_i32;
    dim = (unsigned)dim_i32;

    size_t expected_actual_file_size = npts * dim * data_size + 2 * sizeof(uint32_t);
    if (actual_file_size != expected_actual_file_size)
    {
        std::stringstream stream;
        stream << "Error. File size mismatch. Actual size is " << actual_file_size << " while expected size is  "
               << expected_actual_file_size << " npts = " << npts << " dim = " << dim << " data_size= " << data_size
               << std::endl;
        throw std::runtime_error(stream.str());
    }
    rounded_dim = ROUND_UP(dim, 8);
    LOG_INFO("Metadata: #pts = %lu, #dims = %lu, aligned_dim = %lu... ", npts, dim, rounded_dim);
    size_t allocSize = npts * rounded_dim * data_size;
    LOG_INFO("allocating aligned memory of %lu bytes... ", allocSize);
    alloc_aligned(((void **)&data), allocSize, 8 * data_size);
    LOG_INFO("done. Copying data to mem_aligned buffer... done.");

    for (size_t i = 0; i < npts; i++)
    {
        reader.read(((char *)data + i * rounded_dim * data_size), dim * data_size);
        memset((char *)data + i * rounded_dim * data_size + dim, 0, (rounded_dim - dim) * data_size);
    }
}

inline void load_aligned_bin(uint16_t data_size, const std::string &bin_file, void *&data, size_t &npts, size_t &dim, size_t &rounded_dim)
{
    std::ifstream reader;
    reader.exceptions(std::ifstream::failbit | std::ifstream::badbit);

    try
    {
        LOG_INFO("Reading (with alignment) bin file %s ...", bin_file.c_str());
        reader.open(bin_file, std::ios::binary | std::ios::ate);

        uint64_t fsize = reader.tellg();
        reader.seekg(0);
        load_aligned_bin_impl(data_size, reader, fsize, data, npts, dim, rounded_dim);
    }
    catch (std::system_error &e)
    {
        LOG_ERROR("io_error with load_aligned_file");
        throw e;
    }
}

inline void load_truthset(const std::string &bin_file, uint32_t *&ids, float *&dists, size_t &npts, size_t &dim)
{
    size_t read_blk_size = 64 * 1024 * 1024;
    cached_ifstream reader(bin_file, read_blk_size);
    LOG_INFO("Reading truthset file %s ...", bin_file.c_str());
    size_t actual_file_size = reader.get_file_size();

    int npts_i32, dim_i32;
    reader.read((char *)&npts_i32, sizeof(int));
    reader.read((char *)&dim_i32, sizeof(int));
    npts = (unsigned)npts_i32;
    dim = (unsigned)dim_i32;

    LOG_INFO("Metadata: #pts = %lu, #dims = %lu", npts, dim);

    int truthset_type = -1; // 1 means truthset has ids and distances, 2 means
                            // only ids, -1 is error
    size_t expected_file_size_with_dists = 2 * npts * dim * sizeof(uint32_t) + 2 * sizeof(uint32_t);

    if (actual_file_size == expected_file_size_with_dists)
        truthset_type = 1;

    size_t expected_file_size_just_ids = npts * dim * sizeof(uint32_t) + 2 * sizeof(uint32_t);

    if (actual_file_size == expected_file_size_just_ids)
        truthset_type = 2;

    if (truthset_type == -1)
    {
        std::stringstream stream;
        stream << "Error. File size mismatch. File should have bin format, with "
                  "npts followed by ngt followed by npts*ngt ids and optionally "
                  "followed by npts*ngt distance values; actual size: "
               << actual_file_size << ", expected: " << expected_file_size_with_dists << " or "
               << expected_file_size_just_ids;
        throw std::runtime_error(stream.str());
    }

    ids = new uint32_t[npts * dim];
    reader.read((char *)ids, npts * dim * sizeof(uint32_t));

    if (truthset_type == 1)
    {
        dists = new float[npts * dim];
        reader.read((char *)dists, npts * dim * sizeof(float));
    }
}

template <typename T>
inline size_t save_bin(const std::string &filename, void *data, size_t npts, size_t ndims, size_t offset = 0)
{
    std::ofstream writer;
    open_file_to_write(writer, filename);

    LOG_INFO("Writing bin: %s", filename.c_str());
    writer.seekp(offset, writer.beg);
    int npts_i32 = (int)npts, ndims_i32 = (int)ndims;
    size_t bytes_written = npts * ndims * sizeof(T) + 2 * sizeof(uint32_t);
    writer.write((char *)&npts_i32, sizeof(int));
    writer.write((char *)&ndims_i32, sizeof(int));
    LOG_INFO("bin: #pts = %lu, #dims = %lu, size = %luB", npts, ndims, bytes_written);

    writer.write((char *)data, npts * ndims * sizeof(T));
    writer.close();

    LOG_INFO("Finished writing bin.");
    return bytes_written;
}

inline double calculate_recall(uint32_t num_queries, uint32_t *gold_std, float *gs_dist, uint32_t dim_gs,
                        uint32_t *our_results, uint32_t dim_or, uint32_t recall_at)
{
    double total_recall = 0;
    std::set<uint32_t> gt, res;

    for (size_t i = 0; i < num_queries; i++)
    {
        gt.clear();
        res.clear();
        uint32_t *gt_vec = gold_std + dim_gs * i;
        uint32_t *res_vec = our_results + dim_or * i;
        size_t tie_breaker = recall_at;
        if (gs_dist != nullptr)
        {
            tie_breaker = recall_at - 1;
            float *gt_dist_vec = gs_dist + dim_gs * i;
            while (tie_breaker < dim_gs && gt_dist_vec[tie_breaker] == gt_dist_vec[recall_at - 1])
                tie_breaker++;
        }

        gt.insert(gt_vec, gt_vec + tie_breaker);
        res.insert(res_vec,
                   res_vec + recall_at); // change to recall_at for recall k@k
                                         // or dim_or for k@dim_or
        uint32_t cur_recall = 0;
        for (auto &v : gt)
        {
            if (res.find(v) != res.end())
            {
                cur_recall++;
            }
        }
        total_recall += cur_recall;
    }
    return total_recall / (num_queries) * (100.0 / recall_at);
}

template <typename InType, typename OutType>
inline void convert_types(const InType *srcmat, OutType *destmat, size_t npts, size_t dim)
{
    for (int64_t i = 0; i < (int64_t)npts; i++)
    {
        for (uint64_t j = 0; j < dim; j++)
        {
            destmat[i * dim + j] = (OutType)srcmat[i * dim + j];
        }
    }
}

template <typename T>
inline void print_vector(const T * vector, uint16_t dim) {
    for (uint16_t i = 0; i < dim; i++) {
        std::cout << vector[i] << ", ";
    }
    std::cout << std::endl;
}

inline std::string shell_exec(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

MERCURY_NAMESPACE_END(core);