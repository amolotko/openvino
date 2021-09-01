#include <iostream>
#include <fstream>
#include <vector>
#include "binary_buffer.hpp"
#include "vector_serializer.hpp"

struct A {
    const int Aa;
    double Ab;
    std::vector<double> ss;

    A() : Aa(0) {}
    A(int a, double b, const std::vector<double>& s) : Aa(a), Ab(b), ss(s) {}

    template <typename BT>
    void serialize(BT& b) {
        b(Aa, Ab, ss);
    }
};

int main() {
    int a = 73;
    float b = 3.14f;
    unsigned c = 90;
    double d = 2.72;
    bool e = true;
    const A a_a(11, 78.99, {66.0, 78.09});
    std::vector<int> v{1, 2, 3, 5, 7, 9};
    {
        std::ofstream ofs("archive.bin", std::ios::binary);
        cldnn::BinaryOutputBuffer ob(ofs);

        ob(a, b, c, d, e, v, a_a);
    }

    std::cout << "a: " << a << ", b: " << b << ", c: " << c << ", d: " << d << ", e: " << e << std::endl;
    int aa = 0;
    float bb = .0f;
    unsigned cc = 0;
    const double dd = .4;
    bool ee = false;
    A a_b;
    std::vector<int> vv;
    {
        std::ifstream ifs("archive.bin", std::ios::binary);
        cldnn::BinaryInputBuffer bi(ifs);

        bi(aa, bb, cc, dd, ee, vv, a_b);
    }
    std::cout << "aa: " << aa << ", bb: " << bb << ", cc: " << cc << ", dd: " << dd << ", ee: " << ee << std::endl;
    std::cout << v.size() << " " << vv.size() << std::endl;
    std::cout << a_b.Aa << " " << a_b.Ab << " " << a_b.ss[0] << " " << a_b.ss[1] << std::endl;
    return 0;
}