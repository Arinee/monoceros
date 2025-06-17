/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     gpu_group_ivf_batch_task.h
 *   \author   anduo@xiaohongshu.com
 *   \date     March 2023
 *   \version  1.0.0
 *   \brief    gpu group ivf batch task
 */
#ifdef ENABLE_GPU_IN_MERCURY_

#ifndef INTERFACE_GPU_RESOURCES_WRAPPER_H_
#define INTERFACE_GPU_RESOURCES_WRAPPER_H_

#include "src/core/algorithm/gpu_group_ivf/gpu_neutron_interface.h"
#include "src/core/common/common.h"
#include "src/core/utils/mercury_monitor/singleton.h"

MERCURY_NAMESPACE_BEGIN(core);
class GpuResourcesWrapper : public util::Singleton<GpuResourcesWrapper>
{
    friend class util::Singleton<GpuResourcesWrapper>;

public:
    GpuResourcesWrapper()
    {
        neutron_manager_interface_ = new neutron::gpu::NeutronManagerInterface();
    }

    ~GpuResourcesWrapper()
    {
        std::cout<<"Global GpuResourcesWrapper destructor start"<<std::endl;
        putil::ScopedWriteLock lock(lock_);
        if (neutron_manager_interface_) {
            delete neutron_manager_interface_;
            neutron_manager_interface_ = nullptr;
        }
        std::cout<<"Global GpuResourcesWrapper destructor end"<<std::endl;
    }

    neutron::gpu::NeutronManagerInterface *GetNeutronManagerInterface()
    {
        return neutron_manager_interface_;
    }

    void* GetIndexGpuRecord(std::string& key)
    {  
        putil::ScopedWriteLock lock(lock_);
        if (index_gpu_record_.count(key) > 0) {
            return index_gpu_record_[key].first;
        }
        return nullptr;
    }

    bool IncIndexGpuRecord(std::string& key)
    {
        putil::ScopedWriteLock lock(lock_);
        index_gpu_record_[key].second++;
        return true;
    }

    bool AddIndexGpuRecord(std::string& key, void* gpu_index)
    {
        putil::ScopedWriteLock lock(lock_);
        index_gpu_record_[key] = std::make_pair(gpu_index, 1);
        return true;
    }

    bool RemoveIndexGpuRecord(std::string& key)
    {
        putil::ScopedWriteLock lock(lock_);
        if (--index_gpu_record_[key].second == 0) {
            index_gpu_record_.erase(key);
            return true;
        }
        return false;
    }

private:
    putil::ReadWriteLock lock_;
    neutron::gpu::NeutronManagerInterface *neutron_manager_interface_;
    std::unordered_map<std::string, std::pair<void*, int>> index_gpu_record_;
};

MERCURY_NAMESPACE_END(core);

#endif // INTERFACE_GPU_RESOURCES_WRAPPER_H_
#endif // ENABLE_GPU_IN_MERCURY_