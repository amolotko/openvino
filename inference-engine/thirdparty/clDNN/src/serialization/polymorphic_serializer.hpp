#pragma once

#include <memory>
#include <exception>
#include <type_traits>
#include "buffer.hpp"
#include "bind.hpp"
#include "helpers.hpp"
#include "object_types.hpp"
#include "test_engine.hpp"

namespace cldnn {

template <typename BufferType, typename T>
class Serializer<BufferType, std::unique_ptr<T>, typename std::enable_if<std::is_base_of<OutputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void save(BufferType& buffer, const std::unique_ptr<T>& ptr) {
        const auto& type = ptr->getHash();
        buffer << cldnn::make_data(&type, sizeof(cldnn::PrimitiveImplType));
        const auto save_func = saver_storage<BufferType>::instance().get_save_function(type);
        save_func(buffer, ptr.get());
    }
};

template <typename BufferType, typename T>
class Serializer<BufferType, std::unique_ptr<T>, typename std::enable_if<std::is_base_of<InputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void load(BufferType& buffer, std::unique_ptr<T>& ptr, Engine& engine) {
        cldnn::PrimitiveImplType type;
        buffer >> cldnn::make_data(&type, sizeof(cldnn::PrimitiveImplType));
        const auto load_func = loader_storage<BufferType>::instance().get_load_function(type);
        std::unique_ptr<void, void_deleter<void>> result;
        load_func(buffer, result, engine);
        ptr.reset(static_cast<T*>(result.release()));
    }
};

}