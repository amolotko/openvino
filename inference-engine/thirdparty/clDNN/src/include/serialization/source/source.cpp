#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include "test_classes.hpp"
#include "binary_buffer.hpp"
#include "vector_serializer.hpp"
#include "string_serializer.hpp"
#include "polymorphic_serializer.hpp"
#include "helpers.hpp"


struct A {
    int Aa;
    double Ab;
    std::vector<double> ss;

    A() : Aa(0) {}
    A(int a, double b, const std::vector<double>& s) : Aa(a), Ab(b), ss(s) {}

    void save(cldnn::BinaryOutputBuffer& buffer) const {
        buffer(Aa, Ab, ss);
    }

    void load(cldnn::BinaryInputBuffer& buffer) {
        buffer(Aa, Ab, ss);
    }
};

enum Type : uint64_t {
    T1,
    T2 = 55555555555,
    T3
};

union S
{
    char a;
    short b;
    int c;
    long d;
    double e;
    //A aa;
};


int main() {
    std::cout << "+++START PROGRAM+++" << std::endl;
    int a = 73;
    float b = 3.14f;
    unsigned c = 90;
    const double d = 2.72;
    bool e = true;
    A a_a(11, 78.99, {66.0, 78.09});
    A a1(1, 67.9999, {1.9, 3.6});
    std::vector<A> vA{a_a, a1};
    std::vector<int> v{1, 2, 3, 5, 7, 9};
    const std::string str1 = "qqqq,.,.,";
    Type type = Type::T3;
    S u = {'a'};
    u.e = 4.77;

    Engine engine;
    
    std::unique_ptr<B> d1 = create_D1(engine);
    std::unique_ptr<B> d2 = create_D2(engine);
    {
        std::ofstream ofs("archive.bin", std::ios::binary);
        cldnn::BinaryOutputBuffer ob(ofs);

        ob(a, b, c, d, e, v, a_a, vA, str1, cldnn::make_data(&type, sizeof(Type)), cldnn::make_data(&u, sizeof(S)));
        ob << d1 << d2;
    }

    std::cout << "a: " << a << ", b: " << b << ", c: " << c << ", d: " << d << ", e: " << e << " type: " << type << " u: " << u.e << std::endl;
    int aa = 0;
    float bb = .0f;
    unsigned cc = 0;
    double dd = .4;
    bool ee = false;
    A a_b;
    std::vector<int> vv;
    std::vector<A> vvA;
    std::string str;
    Type type1;
    S u1;
    std::unique_ptr<B> new_d1;
    std::unique_ptr<B> new_d2;
    {
        std::ifstream ifs("archive.bin", std::ios::binary);
        cldnn::BinaryInputBuffer bi(ifs, engine);

        bi(aa, bb, cc, dd, ee, vv, a_b, vvA, str, cldnn::make_data(&type1, sizeof(Type)), cldnn::make_data(&u1, sizeof(S)));
        bi >> new_d1 >> new_d2;
    }
    new_d1->say();
    new_d2->say();
    std::cout << "aa: " << aa << ", bb: " << bb << ", cc: " << cc << ", dd: " << dd << ", ee: " << ee << " type: " << type1 << " ,u1: " << u1.e << std::endl;
    for (size_t i = 0; i < v.size(); i++) {
        std::cout << "v[" << i << "]: " << v[i] << " vv[" << i << "]: " << vv[i] << std::endl;
    }
    std::cout << a_b.Aa << " " << a_b.Ab << " " << a_b.ss[0] << " " << a_b.ss[1] << std::endl;
    for (auto& el : vvA) {
        std::cout <<el.Aa << " " << el.Ab << " " << el.ss[0] << " " << el.ss[1] << std::endl;
    }
    std::cout << str << std::endl;
    return 0;
}