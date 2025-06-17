/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     index_cluster.h
 *   \author   Hechong.xyf
 *   \date     Jun 2018
 *   \version  1.0.0
 *   \brief    Interface of Mercury Index Cluster
 */

#ifndef __MERCURY_INDEX_CLUSTER_H__
#define __MERCURY_INDEX_CLUSTER_H__

#include "index_meta.h"
#include "index_params.h"
#include "utility/thread_pool.h"

MERCURY_NAMESPACE_BEGIN(core);
/*! Index Cluster Parameters
 */
using ClusterParams = IndexParams;

/*! Index Cluster
 */
class IndexCluster
{
public:
    /*! Index Cluster Centroid
     */
    class Centroid
    {
    public:
        //! Constructor
        Centroid(void) : _buffer(), _score(0.0), _follows(0), _similars() {}

        //! Constructor
        Centroid(const void *feat, size_t bytes)
            : _buffer(std::string(reinterpret_cast<const char *>(feat), bytes)),
              _score(0.0), _follows(0)
        {
        }

        //! Constructor
        Centroid(const Centroid &rhs)
            : _buffer(rhs._buffer), _score(rhs._score), _follows(rhs._follows)
        {
        }

        //! Constructor
        Centroid(Centroid &&rhs)
            : _buffer(std::move(rhs._buffer)), _score(rhs._score),
              _follows(rhs._follows)
        {
        }

        //! Assignment
        Centroid &operator=(const Centroid &rhs)
        {
            _buffer = rhs._buffer;
            _score = rhs._score;
            _follows = rhs._follows;
            return *this;
        }

        //! Assignment
        Centroid &operator=(Centroid &&rhs)
        {
            _buffer = std::move(rhs._buffer);
            _score = rhs._score;
            _follows = rhs._follows;
            return *this;
        }

        //! Less than
        bool operator<(const Centroid &rhs) const
        {
            return (this->_score < rhs._score);
        }

        //! Set feature of centroid
        void setFeature(const void *feat, size_t bytes)
        {
            _buffer.assign(
                std::string(reinterpret_cast<const char *>(feat), bytes));
        }

        //! Set score of centroid
        void setScore(double val)
        {
            _score = val;
        }

        //! Set follows of centroid
        void setFollows(size_t count)
        {
            _follows = count;
        }

        //! Set similars of centroid
        void setSimilars(const std::vector<const void *> &feats)
        {
            _similars = feats;
        }

        //! Set similars of centroid
        void setSimilars(std::vector<const void *> &&feats)
        {
            _similars = std::move(feats);
        }

        //! Retrieve feature pointer
        const void *feature(void) const
        {
            return _buffer.data();
        }

        //! Retrieve size of centroid in bytes
        size_t size(void) const
        {
            return _buffer.size();
        }

        //! Retrieve score of centroid
        double score(void) const
        {
            return _score;
        }

        //! Retrieve follows' count of centroid
        size_t follows(void) const
        {
            return _follows;
        }

        //! Retrieve similars of centroid
        const std::vector<const void *> &similars(void) const
        {
            return _similars;
        }

        //! Retrieve similars of centroid
        std::vector<const void *> &similars(void)
        {
            return _similars;
        }

    private:
        //! Members
        std::string _buffer;
        double _score;
        size_t _follows;
        std::vector<const void *> _similars;
    };

    //! Index Cluster Pointer
    typedef std::shared_ptr<IndexCluster> Pointer;

    //! Destructor
    virtual ~IndexCluster(void) {}

    //! Initialize Cluster
    virtual int init(const IndexMeta &meta, const ClusterParams &params) = 0;

    //! Cleanup Cluster
    virtual int cleanup(void) = 0;

    //! Suggest parameters of cluster
    virtual void suggest(const ClusterParams &params) = 0;

    //! Mount features
    virtual int mount(const void *const *feats, size_t count) = 0;

    //! Mount features
    virtual int mount(const void *const *feats, size_t count,
                      size_t samples) = 0;

    //! Mount features (one flat buffer)
    virtual int mount(const void *feats, size_t count) = 0;

    //! Mount features (one flat buffer)
    virtual int mount(const void *feats, size_t count, size_t samples) = 0;

    //! Cluster
    virtual int cluster(ThreadPool &pool, std::vector<Centroid> &cents) = 0;

    //! Classify
    virtual int classify(ThreadPool &pool, std::vector<Centroid> &cents) = 0;

    //! Label
    virtual int label(mercury::ThreadPool &pool, std::vector<Centroid> &cents,
                      std::vector<uint32_t> *out) = 0;
};

MERCURY_NAMESPACE_END(core);
#endif // __MERCURY_INDEX_CLUSTER_H__
