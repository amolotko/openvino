#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "object_types.hpp"
#include "helpers.hpp"

class B {
public:
    B() = default;

    B(const std::string& str) : str(str) {}

    virtual ~B() = default;

    virtual cldnn::PrimitiveImplType getHash() {
        return type;
    }
    virtual void say() {
        std::cout << "----------B::say----------" << std::endl;
        std::cout << str << std::endl;
    }

    template <typename T>
    void save(T& buffer) const {
        buffer << str;
        std::cout << "Save B" << std::endl;
    }

    template <typename T>
    void load(T& buffer) {
        buffer(str);
        std::cout << "Load B" << std::endl;
    }
private:
    static const cldnn::PrimitiveImplType type;
    std::string str;
};


class D1 : public B {
    using parent = B;
public:
    D1() = default;

    D1(const std::vector<std::string> vs) : parent("from D1"), vs(vs) {}

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
        std::cout << "Load D1" << std::endl;
    }
private:
    static const cldnn::PrimitiveImplType type;
    std::vector<std::string> vs;
};

enum class Color {RED, GREEN, BLUE};
class D2 : public B {
    using parent = B;
public:
    D2() = default;

    D2(const Color& color) : parent("from D2"), color(color) {}

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
        std::cout << "Load D2" << std::endl;
    }
private:
    static const cldnn::PrimitiveImplType type;
    Color color = Color::BLUE;
};
