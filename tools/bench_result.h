#ifndef BENCH_RESULT_H 
#define BENCH_RESULT_H 

#include <string.h>
#include <queue>
#include <map>
#include <pthread.h>
#include <sys/time.h>
#include <map>
#include <stdio.h>

class BenchResult
{ 
public:
    BenchResult()
    {
        pthread_spin_init(&_lock, PTHREAD_PROCESS_PRIVATE);
        _totalQueryCount = 0;
        _totalProcessTimeByUs = 0;
    }
    ~BenchResult()
    {
        pthread_spin_destroy(&_lock);
    }

    void addTime(long timeByUs) {
        pthread_spin_lock(&_lock);
        ++_totalQueryCount;
        _totalProcessTimeByUs += timeByUs;
        long timeVal = timeByUs / 100;
        if (_processTimeMap.find(timeVal) != _processTimeMap.end()) {
            ++_processTimeMap[timeVal];
        } else {
            _processTimeMap[timeVal] = 1;
        }
        pthread_spin_unlock(&_lock);
    }
    void markStart() {
        gettimeofday(&_start, NULL);
    }
    void markEnd() {
        gettimeofday(&_end, NULL);
    }
    long getDurationByMs() {
        long duration = (_end.tv_sec - _start.tv_sec) * 1000 + (_end.tv_usec - _start.tv_usec) / 1000;
        return duration;
    }
    long getTotalQueryCount() {
        return _totalQueryCount;
    }
    std::map<long, long>& getProcessTimeMap() {
        return _processTimeMap;
    }
    long getTotalProcessTimeByMs() {
        return _totalProcessTimeByUs / 1000;
    }
    void print() {
        fprintf(stdout, "Process query: %ld, total process time: %ld ms, duration: %ld ms\n", 
                getTotalQueryCount(), 
                getTotalProcessTimeByMs(), 
                getDurationByMs());
        fprintf(stdout, "Avg latency: %0.1f qps: %0.1f\n", 
                ((float)getTotalProcessTimeByMs())/getTotalQueryCount(),
                getTotalQueryCount()/((float)getDurationByMs()/1000));

        int totNum = 0;
        int percent[] = {25, 50, 75, 90, 95, 99};
        int index = 0;
        float maxTime = 0.0;
        int lastNum = 0;
        for (int i = 0; index < 6 && i < 50000; i++) {
            totNum += _processTimeMap[i];
            if (totNum > 0 && totNum >= (_totalQueryCount)/100*percent[index]) {
                if(lastNum != totNum) {
                    maxTime = (float)i / 10;
                    lastNum = totNum; 
                }
                fprintf(stdout, "%d Percentile:\t\t %.1f ms\n", percent[index], maxTime);
                index ++;
                if (index > 5) {
                    break;
                }
            }
        }
        for (; index < 6; index++) {
            fprintf(stdout, "%d Percentile:\t\t %.1f ms\n", percent[index], maxTime);
        }
        fprintf(stdout, "\n");
    }
private:
    long _totalQueryCount;
    long _totalProcessTimeByUs;
    struct timeval _start;
    struct timeval _end;
    pthread_spinlock_t _lock;
    std::map<long, long> _processTimeMap; // <processTimeBy100us, count>
};

#endif 
