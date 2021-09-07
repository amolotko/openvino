#pragma once

#include <type_traits>
#include <unordered_map>
#include <functional>
#include "buffer.hpp"
#include "singleton.hpp"
#include "object_types.hpp"
#include "test_classes.hpp"

#define BIND_TO_BUFFER(...) \
        template <> \
        class bind_creator<__VA_ARGS__> { \
        private: \
            static const instance_creator<__VA_ARGS__>& creator; \
        }; \
        const instance_creator<__VA_ARGS__>& bind_creator<__VA_ARGS__>::creator = Singleton<instance_creator<__VA_ARGS__>>::getInstance().instantiate();
namespace cldnn {

template <typename BufferType, typename Enable = void>
class buffer_binder {};

template <typename BufferType>
class buffer_binder<BufferType, typename std::enable_if<std::is_base_of<OutputBuffer<BufferType>, BufferType>::value>::type> {
public:
    std::function<void(BufferType&, const void*)> getSaveFunction(const PrimitiveImplType& type) const {
        return save_functions.at(type);
    }

    buffer_binder() {
        save_functions[PrimitiveImplType::B] = (*this).template save<B>;
        save_functions[PrimitiveImplType::D1] = (*this).template save<D1>;
        save_functions[PrimitiveImplType::D2] = (*this).template save<D2>;
    }
private:

    template <typename Derived>
    static const Derived* downcast(const void* base_ptr) {
        return static_cast<Derived const *>(base_ptr);
    }

    template <typename T>
    static void save(BufferType& buffer, const void* base_ptr) {
        const auto derived_ptr = downcast<T>(base_ptr);
        derived_ptr->save(buffer);
    }

    std::unordered_map<PrimitiveImplType, std::function<void(BufferType&, const void*)>> save_functions;
};

template <typename T>
struct void_deleter {
    void operator()(const T*) const { }
};

template <typename BufferType>
class buffer_binder<BufferType, typename std::enable_if<std::is_base_of<InputBuffer<BufferType>, BufferType>::value>::type> {
public:
    buffer_binder() {
        load_functions[PrimitiveImplType::B] = (*this).template load<B>;
        load_functions[PrimitiveImplType::D1] = (*this).template load<D1>;
        load_functions[PrimitiveImplType::D2] = (*this).template load<D2>;
    }

    std::function<void(BufferType&, std::unique_ptr<void, void_deleter<void>>&)> getLoadFunction(const PrimitiveImplType& type) const {
        return load_functions.at(type);
    }
private:
    template <typename T>
    static void load(BufferType& buffer, std::unique_ptr<void, void_deleter<void>>& result_ptr) {
        std::unique_ptr<T> derived_ptr = std::unique_ptr<T>(new T());
        derived_ptr->load(buffer);
        result_ptr.reset(derived_ptr.release());
    }

    std::unordered_map<PrimitiveImplType, std::function<void(BufferType&, std::unique_ptr<void, void_deleter<void>>&)>> load_functions;
};

template <typename T>
class bind_creator;

template <typename BufferType>
class instance_creator {
public:
    const instance_creator& instantiate() {
        Singleton<buffer_binder<BufferType>>::getInstance();
        return *this;
    }
};

}
