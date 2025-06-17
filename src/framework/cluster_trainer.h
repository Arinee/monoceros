/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     trainer.h
 *   \author   qiuming
 *   \date     Jan 2019
 *   \version  1.0.0
 *   \brief    Cluster Trainer
 */

#ifndef __MERCURY_CLUSTER_TRAINER_H__
#define __MERCURY_CLUSTER_TRAINER_H__

namespace mercury 
{

class IndexParams;
class IndexMeta;
class IndexStorage;
class VectorHolder;

/*! Cluster Trainer
 */
class ClusterTrainer
{
public:
    //! Cluster Trainer Pointer
    typedef std::shared_ptr<ClusterTrainer> Pointer;

    //! Destructor
    virtual ~ClusterTrainer(void) {}

    //! Initialize Trainer 
    virtual int Init(const IndexMeta &meta, const IndexParams &params) = 0;

    //! Cleanup Trainer 
    virtual int Cleanup(void) = 0;

    //! Train the data
    virtual int Train(const VectorHolder &holder) = 0;

    //! Dump index into file or memory
    virtual int DumpIndex(const std::string &path, const IndexStorage &stg) = 0;

protected:
    //! Init Centroids
    virtual int InitCentroids() = 0;

    //! Update Clusters
    virtual int updateClusters() = 0;

    //! Update Centroids
    virtual int updateCentroids() = 0;
};

} // namespace mercury

#endif // __MERCURY_CLUSTER_TRAINER_H__
