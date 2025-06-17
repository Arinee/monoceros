#pragma once

#include "../centroid_resource.h"
#include "src/core/common/common.h"
#include "src/core/common/pq_common.h"
#include "src/core/framework/index_meta.h"
#include <assert.h>
#include <memory>

MERCURY_NAMESPACE_BEGIN(core);

class LoopUpTable
{
public:
    typedef std::shared_ptr<LoopUpTable> Pointer;
    typedef std::priority_queue<CentroidInfo, std::vector<CentroidInfo>, std::greater<CentroidInfo>> CentroidCandidate;

    LoopUpTable();
    LoopUpTable(const IndexMeta &indexMeta, CentroidResource *centroidResource);
    ~LoopUpTable();

public:
    bool init(const void *queryFeature, const std::vector<size_t> &levelScanLimit, bool withCodeFeature = false);
    bool initLoopUpTable(const void *queryFeature, bool withCodeFeature = false);
    // only for build and label
    bool getQueryCodeFeature(UInt8Vector &queryCodeFeature);

    void *GetDistanceArray()
    {
        return _distanceArray;
    }

    void *GetQDistanceArray()
    {
        return _qdistanceArray;
    }

    distance_t getDistance(size_t centroidIndex, size_t fragmentIndex) const
    {
        assert(fragmentIndex < _fragmentNum);
        assert(centroidIndex < _centroidNum);
        return _distanceArray[fragmentIndex * _centroidNum + centroidIndex];
    }

    CentroidCandidate &getCentroids()
    {
        return _centroids;
    }

    size_t getFragmentNum() const
    {
        return _fragmentNum;
    }

    float_t getScale() const
    {
        if (_normalizers) {
            return _normalizers[0];
        }
        return 0.0f;
    }

    float_t getBias() const
    {
        if (_normalizers) {
            return _normalizers[1];
        }
        return 0.0f;
    }

    size_t getCentroidNum() const
    {
        return _centroidNum;
    }

    void setWithCodeFeature(bool withCodeFeature)
    {
        this->_withCodeFeature = withCodeFeature;
    }

    inline void print_tab() {
        std::cout << "original distance matrix:" << std::endl;
        for (size_t i = 0; i < _fragmentNum; i++) {
            std::cout << "fragment " << i << ": ";
            for (size_t n = 0; n < _centroidNum; ++n) {
                std::cout << _distanceArray[i * _centroidNum + n] << " ";
            }
            std::cout << std::endl;
        }
    }

    inline void print_qtab() {
        std::cout << "quantized distance matrix:" << std::endl;
        for (size_t i = 0; i < _fragmentNum; i++) {
            std::cout << "fragment " << i << ": ";
            for (size_t n = 0; n < _centroidNum; ++n) {
                std::cout << unsigned(_qdistanceArray[i * _centroidNum + n]) << " ";
            }
            std::cout << std::endl;
        }
    }

    bool computeCentroid(const void *feature, const std::vector<size_t> &levelScanLimit);
    bool computeLoopUpTable(const void *feature);
    bool quantizeLoopUpTable();
    bool initQueryCodeFeature(void);

private:
    CentroidResource *_centroidResource;
    IndexMeta _indexMeta;
    IndexMeta _fragmentIndexMeta;
    uint8_t *_queryCodeFeature;
    CentroidCandidate _centroids;
    bool _withCodeFeature;

    inline float tab_min(const float* tab, size_t n) {
        float min = HUGE_VAL;
        for (size_t i = 0; i < n; i++) {
            if (tab[i] < min)
                min = tab[i];
        }
        return min;
    }

    inline float tab_max(const float* tab, size_t n) {
        float max = -HUGE_VAL;
        for (size_t i = 0; i < n; i++) {
            if (tab[i] > max)
                max = tab[i];
        }
        return max;
    }

    template <typename T>
    inline void round_tab(const float* tab, size_t n, float a, float bi, T* tab_out) {
        for (size_t i = 0; i < n; i++) {
            tab_out[i] = (T)floorf((tab[i] - bi) * a + 0.5);
        }
    }

protected:
    distance_t *_distanceArray;
    q_distance_t *_qdistanceArray;
    float_t *_normalizers;
    size_t _fragmentNum;
    size_t _centroidNum;
    size_t _elemSize;
};

MERCURY_NAMESPACE_END(core);
