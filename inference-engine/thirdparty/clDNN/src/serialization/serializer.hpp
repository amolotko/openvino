#pragma once

namespace cldnn {
    template <typename BufferType, typename T, typename Enable = void>
    class Serializer {
    public:
        static void serialize(BufferType& buffer, T& object) {
            object.serialize(buffer);
        }
    };
}