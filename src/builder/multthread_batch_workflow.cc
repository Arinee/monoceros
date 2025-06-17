#include "multthread_batch_workflow.h"
#include "framework/utility/time_helper.h"

using namespace std;
using namespace mercury;

MuitThreadBatchWorkflow::~MuitThreadBatchWorkflow(void){
}

void MuitThreadBatchWorkflow::Init(const std::function<bool(void)>& finish_task, 
        const std::function<std::list<Closure::Pointer>(void)>& splist_task, 
        const Closure::Pointer& merge_task)
{
    finish_task_ = finish_task;
    splist_task_ = splist_task;
    merge_task_ = merge_task;
}

void MuitThreadBatchWorkflow::Stop(void)
{   
    pool_->stop();
}

bool MuitThreadBatchWorkflow::IsFinish(void)
{
    if(pool_->isFinished()){
        return false;
    }
    return true;
}

void MuitThreadBatchWorkflow::PushBack(const Closure::Pointer &task)
{
    pool_->enqueue(task);
}

void MuitThreadBatchWorkflow::WaitForAllDone(void)
{
    pool_->waitFinish();
}

void MuitThreadBatchWorkflow::Run()
{
    if(!splist_task_){
        LOG_ERROR("MuitThreadBatchWorkflow Init Failed!");
        return;
    }
    //judge whether finished
    bool finished = false;
    while(!finished)
    {
        //do splist
        ElapsedTime elapsed_time;
        std::list<Closure::Pointer> task_list = splist_task_();
        LOG_ERROR("splist time cose:%ld", elapsed_time.elapsed());
        //submit all task
        LOG_ERROR("Work Dispatch %ld", task_list.size());
        elapsed_time.update();
        while (!task_list.empty()) {
            auto task = task_list.front();
            pool_->enqueue(task);
            task_list.pop_front();
        }
        LOG_ERROR("enqueue time cose:%ld", elapsed_time.elapsed());
        //submit all task
        pool_->wakeAll();
        WaitForAllDone();
        LOG_ERROR("Work All Done");
        //do merge
        if(merge_task_){
            merge_task_->run();
        }

        if(finish_task_)
        {
            finished = finish_task_();
        }else{
            finished = true;
        }
    }
}
