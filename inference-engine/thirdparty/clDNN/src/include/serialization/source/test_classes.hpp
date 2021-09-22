#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "object_types.hpp"
#include "helpers.hpp"
#include "string_serializer.hpp"
#include "test_engine.hpp"

class B {
public:
    B(Engine& engine) : engine(engine) {}

    B(Engine& engine, const std::string& str) : engine(engine), str(str) {}

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

    Engine& getEngine() {
        return engine;
    }

    virtual void abstract() = 0;

    static const cldnn::PrimitiveImplType type;
protected:
    Engine& engine;
private:
    std::string str;
};

std::unique_ptr<B> create_D1(Engine& engine);
std::unique_ptr<B> create_D2(Engine& engine);