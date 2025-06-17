/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     multthread_batch_workflow.h
 *   \author   yunheng@xiaohongshu.com
 *   \date     Jan 2019
 *   \version  1.0.0
 *   \brief    Threadable WorkFlow
 */

#ifndef __MERCURY_THREADABLE_BATCH_WORKFLOW_H__
#define __MERCURY_THREADABLE_BATCH_WORKFLOW_H__

#include <memory>
#include <atomic>
#include <functional>
#include <list>
#include "framework/index_logger.h"
#include "framework/utility/closure.h"
#include "framework/utility/mmap_file.h"
#include "framework/utility/thread_pool.h"

namespace mercury 
{

/*! thrad workflow batch process
 */
class MuitThreadBatchWorkflow
{
public:
    //! MuitThreadBatchWorkflow Pointer
    typedef std::shared_ptr<MuitThreadBatchWorkflow> Pointer;

    //! Constructor
    MuitThreadBatchWorkflow(int thread_num):state_(0),thread_num_(thread_num)
    {
        if (thread_num_ > 0) {
            pool_.reset(new mercury::ThreadPool(false, thread_num_));
        } else {
            pool_.reset(new mercury::ThreadPool);
        }   
    }
    
    //! Destructor
    ~MuitThreadBatchWorkflow(void);

    //! Init
    void Init(const std::function<bool(void)>& finish_task, 
        const std::function<std::list<Closure::Pointer>(void)>& splist_task, 
        const Closure::Pointer& merge_task = Closure::Pointer());

    //! Stop Thread
    void Stop(void);

    //! Judge State
    bool IsFinish(void);

    //! Push Job
    void PushBack(const Closure::Pointer &task);

    //! Wait All Submited Task Done
    void WaitForAllDone(void);
    
    //! Do Batch MR Job
    void Run();
private:
    int state_;
    int thread_num_;
    std::shared_ptr<mercury::ThreadPool> pool_;

    //new binded fucntion
    std::function<bool(void)> finish_task_;
    std::function<std::list<Closure::Pointer>(void)> splist_task_;
    Closure::Pointer merge_task_;
};

} // namespace mercury

#endif // __MERCURY_THREADABLE_BATCH_WORKFLOW_H__
