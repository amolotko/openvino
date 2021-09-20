#include "test_classes.hpp"

enum class Color {RED, GREEN, BLUE};
class D2 : public B {
    using parent = B;
public:
    D2(Engine& engine) : parent(engine) {}

    D2(Engine& engine, const Color& color) : parent(engine, "from D2"), color(color) {}

    cldnn::PrimitiveImplType getHash() override {
        return type;
    }

    void say() override {
        parent::say();
        std::cout << "----------D2----------" << std::endl;
        std::cout << static_cast<std::size_t>(color) << std::endl;
    }

    template <typename T>
    void save(T& buffer) const {
        parent::save(buffer);
        buffer << cldnn::make_data(&color, sizeof(Color));
        std::cout << "Save D2" << std::endl;
    }

    template <typename T>
    void load(T& buffer) {
        parent::load(buffer);
        buffer(cldnn::make_data(&color, sizeof(Color)));
        engine.allocate();
        std::cout << "Load D2" << std::endl;
    }

    void abstract() override {}

    static const cldnn::PrimitiveImplType type;
private:
    Color color = Color::BLUE;
};

const cldnn::PrimitiveImplType D2::type = cldnn::PrimitiveImplType::D2;

std::unique_ptr<B> create_D2(Engine& engine) {
    return std::make_unique<D2>(engine, Color::GREEN);
}

#include "binary_buffer.hpp"

BIND_BINARY_BUFFER_WITH_TYPE(D2)