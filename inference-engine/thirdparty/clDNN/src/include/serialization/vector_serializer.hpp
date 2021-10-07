#pragma once

#include <vector>
#include <type_traits>
#include "buffer.hpp"
#include "helpers.hpp"
#include "kernel_selector_common.h"
#include "cldnn/runtime/kernel_args.hpp"

namespace cldnn {
template <typename BufferType, typename T>
class Serializer<BufferType, std::vector<T>, typename std::enable_if<std::is_base_of<OutputBuffer<BufferType>, BufferType>::value &&
                                                                     std::is_arithmetic<T>::value &&
                                                                    !std::is_same<bool, T>::value>::type> {
public:
    static void save(BufferType& buffer, const std::vector<T>& vector) {
        buffer << vector.size(); //static_cast<uint64_t>()
        buffer << make_data(vector.data(), static_cast<uint64_t>(vector.size() * sizeof(T)));
    }
};

template <typename BufferType, typename T>
class Serializer<BufferType, std::vector<T>, typename std::enable_if<std::is_base_of<InputBuffer<BufferType>, BufferType>::value &&
                                                                     std::is_arithmetic<T>::value &&
                                                                     !std::is_same<bool, T>::value>::type> {
public:
    static void load(BufferType& buffer, std::vector<T>& vector) {
        typename std::vector<T>::size_type vector_size = 0UL;
        buffer >> vector_size;
        vector.resize(vector_size);
        buffer >> make_data(vector.data(), static_cast<uint64_t>(vector_size * sizeof(T)));
    }
};

template <typename BufferType, typename T>
class Serializer<BufferType, std::vector<T>, typename std::enable_if<std::is_base_of<OutputBuffer<BufferType>, BufferType>::value &&
                                                                    !std::is_arithmetic<T>::value &&
                                                                    !std::is_same<kernel_selector::clKernelData, T>::value>::type> {
public:
    static void save(BufferType& buffer, const std::vector<T>& vector) {
        buffer << vector.size();
        for (const auto& el : vector) {
            buffer << el;
        }
    }
};

template <typename BufferType, typename T>
class Serializer<BufferType, std::vector<T>, typename std::enable_if<std::is_base_of<InputBuffer<BufferType>, BufferType>::value &&
                                                                    !std::is_arithmetic<T>::value &&
                                                                    !std::is_same<kernel_selector::clKernelData, T>::value>::type> {
public:
    static void load(BufferType& buffer, std::vector<T>& vector) {
        typename std::vector<T>::size_type vector_size = 0UL;
        buffer >> vector_size;
        vector.resize(vector_size);
        for (auto& el : vector) {
            buffer >> el;
        }
    }
};

template <typename BufferType, typename T>
class Serializer<BufferType, std::vector<T>, typename std::enable_if<std::is_base_of<OutputBuffer<BufferType>, BufferType>::value &&
                                                                     std::is_same<kernel_selector::clKernelData, T>::value>::type> {
public:
    static void save(BufferType& buffer, const std::vector<T>& vector) {
        buffer << vector.size();
        for (const auto& el : vector) {
            const auto& params = el.params;
            buffer << params.workGroups.global << params.workGroups.local;
            buffer << params.arguments.size();
            for (const auto& arg : params.arguments) {
                buffer << make_data(&arg.t, sizeof(argument_desc::Types)) << arg.index;
            }
            buffer << params.scalars.size();
            for (const auto& scalar : params.scalars) {
                buffer << make_data(&scalar.t, sizeof(scalar_desc::Types)) << make_data(&scalar.v, sizeof(scalar_desc::ValueT));
            }
            buffer << params.layerID;
        }
    }
};

template <typename BufferType, typename T>
class Serializer<BufferType, T, typename std::enable_if<std::is_base_of<InputBuffer<BufferType>, BufferType>::value &&
                                                        std::is_same<kernel_selector::clKernelData, T>::value>::type> {
public:
    static void load(BufferType& buffer, std::vector<T>& vector) {
        typename std::vector<T>::size_type vector_size = 0UL;
        buffer >> vector_size;
        vector.resize(vector_size);
        for (auto& el : vector) {
            // el.code{};
            auto& params = el.params;
            buffer >> params.workGroups.global >> params.workGroups.local;

            typename arguments_desc::size_type arguments_desc_size = 0UL;
            buffer >> arguments_desc_size;
            params.arguments.resize(arguments_desc_size);
            for (auto& arg : params.arguments) {
                buffer >> make_data(&arg.t, sizeof(argument_desc::Types)) >> arg.index;
            }

            typename scalars_desc::size_type scalars_desc_size = 0UL;
            buffer >> scalars_desc_size;
            params.scalars.resize(scalars_desc_size);
            for (auto& scalar : params.scalars) {
                 buffer >> make_data(&scalar.t, sizeof(scalar_desc::Types)) >> make_data(&scalar.v, sizeof(scalar_desc::ValueT));
            }

            buffer >> params.layerID;
        }
    }
};

}