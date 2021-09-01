#pragma once
#include <utility>
#include <type_traits>
#include "serializer.hpp"

namespace cldnn {

template <typename BufferType>
class Buffer {
public:
    Buffer(BufferType* const buffer) : buffer(buffer) {}

    template <typename ... Types>
    inline BufferType& operator()(Types&& ... args) {
        buffer->process(std::forward<Types>(args)...);
        return *buffer;
    }

protected:
    template <typename T>
    inline void process(T&& object) {
        serialize_impl(object);
    }

    BufferType* const buffer;

private:
    template <typename T, typename ... OtherTypes>
    inline void process(T&& first, OtherTypes&& ... remains) {
        buffer->process(std::forward<T>(first));
        buffer->process(std::forward<OtherTypes>(remains)...);
    }

    template <typename T>
    void serialize_impl(const T& obj) {
        Serializer<BufferType, typename std::remove_reference<T>::type>::serialize(*buffer, const_cast<T&>(obj));
    }

};

template <typename BufferType>
class OutputBuffer : public Buffer<BufferType> {
public:
    OutputBuffer(BufferType* const buffer) : Buffer<BufferType>(buffer) {}

    template <typename T>
    inline BufferType& operator<<(T&& arg) {
        Buffer<BufferType>::buffer->process(std::forward<T>(arg));
        return *Buffer<BufferType>::buffer;
    }
};

template <typename BufferType>
class InputBuffer : public Buffer<BufferType> {
public:
    InputBuffer(BufferType* const buffer) : Buffer<BufferType>(buffer) {}

    template <typename T>
    inline BufferType& operator>>(T&& arg) {
        Buffer<BufferType>::buffer->process(std::forward<T>(arg));
        return *Buffer<BufferType>::buffer;
    }
};
}