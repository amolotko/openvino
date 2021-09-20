#include "test_classes.hpp"
#include "vector_serializer.hpp"

class D1 : public B {
    using parent = B;
public:
    D1(Engine& engine) : parent(engine) {}

    D1(Engine& engine, const std::vector<std::string> vs) : parent(engine, "from D1"), vs(vs) {}

    cldnn::PrimitiveImplType getHash() override {
        return type;
    }

    void say() override {
        parent::say();
        std::cout << "----------D1::say----------" << std::endl;
        for (const auto& e : vs) {
            std::cout << e << std::endl;
        }
    }

    template <typename T>
    void save(T& buffer) const {
        parent::save(buffer);
        buffer << vs;
        std::cout << "Save D1" << std::endl;
    }

    template <typename T>
    void load(T& buffer) {
        parent::load(buffer);
        buffer(vs);
        engine.allocate();
        std::cout << "Load D1" << std::endl;
    }

    void abstract() override {}

    static const cldnn::PrimitiveImplType type;
private:
    std::vector<std::string> vs;
};

const cldnn::PrimitiveImplType D1::type = cldnn::PrimitiveImplType::D1;

std::unique_ptr<B> create_D1(Engine& engine) {
    std::vector<std::string> vs{"qqq", "aaa", "zzz"};
    return std::make_unique<D1>(engine, std::move(vs));
}

#include "binary_buffer.hpp"

BIND_BINARY_BUFFER_WITH_TYPE(D1)