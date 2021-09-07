#pragma once

namespace cldnn {

template <typename T>
class Singleton {
public:
    static T& getInstance() {
        return instantiate();
    }
private:
    static T& instantiate() {
        static T singleton;
        (void)instance;
        return singleton;
    }

    static const T& instance;
};

template <typename T>
const T& Singleton<T>::instance = Singleton<T>::instantiate();

}