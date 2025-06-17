#ifndef MERCURY_UTIL_SINGLETON_H
#define MERCURY_UTIL_SINGLETON_H

#include <memory>
#include "putil/Lock.h"
#include "src/core/common/common.h"

namespace mercury {
namespace util {

class LazyInstantiation
{
protected:
    template <typename T>
    static void Create(T *&ptr)
    {
        ptr = new T;
        static std::shared_ptr<T> destroyer(ptr);
    }
};

template <typename T, typename InstPolicy = LazyInstantiation>
class Singleton : private InstPolicy
{
protected:
    Singleton(const Singleton &) {}
    Singleton() {}

public:
    ~Singleton() {}

public:
    /**
     * Provide access to the single instance through double-checked locking
     *
     * @return the single instance of object.
     */
    static T *GetInstance();
};

template <typename T, typename InstPolicy>
inline T *Singleton<T, InstPolicy>::GetInstance()
{
    static T *ptr = 0;
    static putil::RecursiveThreadMutex gLock;

    if (UNLIKELY(!ptr)) {
        putil::ScopedLock sl(gLock);
        if (!ptr) {
            InstPolicy::Create(ptr);
        }
    }
    return const_cast<T *>(ptr);
}

};
};

#endif // MERCURY_UTIL_SINGLETON_H


/* Local Variables: */
/* tab-width: 4 */
/* indent-tabs-mode: nil */
/* coding: utf-8-unix */
/* mode: c++ */
/* End: */
// vim: set tabstop=4 expandtab fileencoding=utf-8 ff=unix ft=cpp norl :
