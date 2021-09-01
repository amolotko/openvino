#pragma once

#include <vector>
#include <type_traits>
#include "buffer.hpp"
#include "serializer.hpp"

namespace cldnn {
template <typename BufferType, typename T>
class Serializer<BufferType, std::vector<T>, typename std::enable_if<std::is_base_of<OutputBuffer<BufferType>, BufferType>::value &&
                                                                     std::is_arithmetic<T>::value &&
                                                                    !std::is_same<bool, T>::value>::type> {
public:
    static void save(BufferType& buffer, const std::vector<T>& vector) {
        buffer << vector.size(); //static_cast<uint64_t>()
        std::cout << "write vector size: " << vector.size() << std::endl;
        // buffer << binary_data(vector.data(), vector.size() * sizeof(T));
    }
};

template <typename BufferType, typename T>
class Serializer<BufferType, std::vector<T>, typename std::enable_if<std::is_base_of<InputBuffer<BufferType>, BufferType>::value &&
                                                                     std::is_arithmetic<T>::value &&
                                                                     !std::is_same<bool, T>::value>::type> {
public:
    static void load(BufferType& buffer, std::vector<T>& vector) {
        std::size_t vector_size;
        buffer >> vector_size;
        vector.resize(vector_size);
        std::cout << "Read vector size: " << vector_size << std::endl;
        // buffer >> binary_data(vector.data(), vector_size * sizeof(T));
    }
};

template <typename BufferType, typename T>
class Serializer<BufferType, std::vector<T>, typename std::enable_if<std::is_base_of<OutputBuffer<BufferType>, BufferType>::value &&
                                                                    !std::is_arithmetic<T>::value>::type> {
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
                                                                    !std::is_arithmetic<T>::value>::type> {
public:
    static void load(BufferType& buffer, std::vector<T>& vector) {
        std::size_t vector_size = 0UL;
        buffer >> vector_size;
        vector.resize(vector_size);
        for (auto& el : vector) {
            buffer >> el;
        }
    }
};
}