/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     instance_factory.h
 *   \author   qiuming@xiaohongshu.com
 *   \date     Mar 2018
 *   \version  1.0.0
 *   \brief    Interface of Instance Factory
 */

#ifndef __MERCURY_INSTANCE_FACTORY_H__
#define __MERCURY_INSTANCE_FACTORY_H__

#include "index_cluster.h"
#include "index_logger.h"
#include "index_storage.h"
#include "vector_holder.h"
#include "index_reformer.h"
#include "vector_service.h"
#include "utility/factory.h"

MERCURY_NAMESPACE_BEGIN(core);
/*! Instance Factory
 */
struct InstanceFactory
{
    //! Create a index logger by name
    static IndexLogger::Pointer CreateLogger(const std::string &name);

    //! Test if the logger is exist
    static bool HasLogger(const std::string &name);

    //! Create a index holder by name
    static VectorHolder::Pointer CreateHolder(const std::string &name);

    //! Test if the holder is exist
    static bool HasHolder(const std::string &name);

    //! Create a index storage by name
    static IndexStorage::Pointer CreateStorage(const std::string &name);

    //! Test if the storage is exist
    static bool HasStorage(const std::string &name);

    //! Test if the builder is exist
    static bool HasBuilder(const std::string &name);

    //! Test if the reducer is exist
    static bool HasMerger(const std::string &name);

    //! Create a Service by name
    static VectorService::Pointer CreateService(const std::string &name);

    //! Test if the searcher is exist
    static bool HasService(const std::string &name);

    //! Create a index cluster by name
    static IndexCluster::Pointer CreateCluster(const std::string &name);

    //! Test if the cluster is exist
    static bool HasCluster(const std::string &name);

    //! Create a index reformer by name
    static IndexReformer::Pointer CreateReformer(const std::string &name);

    //! Test if the reformer is exist
    static bool HasReformer(const std::string &name);
};

//! Register Index Logger
#define INSTANCE_FACTORY_REGISTER_LOGGER(__IMPL__)                                \
    MERCURY_FACTORY_REGISTER(#__IMPL__, mercury::core::IndexLogger, __IMPL__)

//! Register Index Holder
#define INSTANCE_FACTORY_REGISTER_HOLDER(__IMPL__)                                \
    MERCURY_FACTORY_REGISTER(#__IMPL__, mercury::core::VectorHolder, __IMPL__)

//! Register Index Builder
#define INSTANCE_FACTORY_REGISTER_BUILDER(__IMPL__)                               \
    MERCURY_FACTORY_REGISTER(#__IMPL__, mercury::core::IndexBuilder, __IMPL__)

//! Register Vector Service
#define INSTANCE_FACTORY_REGISTER_SEARCHER(__IMPL__)                              \
    MERCURY_FACTORY_REGISTER(#__IMPL__, mercury::core::VectorService, __IMPL__)

//! Register Index Reducer
#define INSTANCE_FACTORY_REGISTER_REDUCER(__IMPL__)                               \
    MERCURY_FACTORY_REGISTER(#__IMPL__, mercury::core::IndexReducer, __IMPL__)

//! Register Index Storage
#define INSTANCE_FACTORY_REGISTER_STORAGE(__IMPL__)                               \
    MERCURY_FACTORY_REGISTER(#__IMPL__, mercury::core::IndexStorage, __IMPL__)

//! Register Index Cluster
#define INSTANCE_FACTORY_REGISTER_CLUSTER(__IMPL__)                               \
    MERCURY_FACTORY_REGISTER(#__IMPL__, mercury::core::IndexCluster, __IMPL__)

//! Register Index Reformer
#define INSTANCE_FACTORY_REGISTER_REFORMER(__IMPL__)                              \
    MERCURY_FACTORY_REGISTER(#__IMPL__, mercury::core::IndexReformer, __IMPL__)

MERCURY_NAMESPACE_END(core);
#endif // __MERCURY_INSTANCE_FACTORY_H__
