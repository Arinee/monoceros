/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     instance_factory.cc
 *   \author   Hechong.xyf
 *   \date     Mar 2018
 *   \version  1.0.0
 *   \brief    Implementation of Instance Factory
 */

#include "instance_factory.h"

namespace mercury {

IndexLogger::Pointer InstanceFactory::CreateLogger(const std::string &name)
{
    return Factory<IndexLogger>::MakeShared(name.c_str());
}

bool InstanceFactory::HasLogger(const std::string &name)
{
    return Factory<IndexLogger>::Has(name.c_str());
}

VectorHolder::Pointer InstanceFactory::CreateHolder(const std::string &name)
{
    return Factory<VectorHolder>::MakeShared(name.c_str());
}

bool InstanceFactory::HasHolder(const std::string &name)
{
    return Factory<VectorHolder>::Has(name.c_str());
}

IndexStorage::Pointer InstanceFactory::CreateStorage(const std::string &name)
{
    return Factory<IndexStorage>::MakeShared(name.c_str());
}

bool InstanceFactory::HasStorage(const std::string &name)
{
    return Factory<IndexStorage>::Has(name.c_str());
}

IndexBuilder::Pointer InstanceFactory::CreateBuilder(const std::string &name)
{
    return Factory<IndexBuilder>::MakeShared(name.c_str());
}

bool InstanceFactory::HasBuilder(const std::string &name)
{
    return Factory<IndexBuilder>::Has(name.c_str());
}

IndexMerger::Pointer InstanceFactory::CreateMerger(const std::string &name)
{
    return Factory<IndexMerger>::MakeShared(name.c_str());
}

bool InstanceFactory::HasMerger(const std::string &name)
{
    return Factory<IndexMerger>::Has(name.c_str());
}

VectorService::Pointer InstanceFactory::CreateService(const std::string &name)
{
    return Factory<VectorService>::MakeShared(name.c_str());
}

bool InstanceFactory::HasService(const std::string &name)
{
    return Factory<VectorService>::Has(name.c_str());
}

IndexCluster::Pointer InstanceFactory::CreateCluster(const std::string &name)
{
    return Factory<IndexCluster>::MakeShared(name.c_str());
}

bool InstanceFactory::HasCluster(const std::string &name)
{
    return Factory<IndexCluster>::Has(name.c_str());
}

IndexReformer::Pointer InstanceFactory::CreateReformer(const std::string &name)
{
    return Factory<IndexReformer>::MakeShared(name.c_str());
}

bool InstanceFactory::HasReformer(const std::string &name)
{
    return Factory<IndexReformer>::Has(name.c_str());
}

} // namespace mercury
