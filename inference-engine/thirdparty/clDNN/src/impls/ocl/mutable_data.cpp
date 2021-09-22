// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mutable_data_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "object_types.hpp"
#include "serialization/binary_buffer.hpp"

namespace cldnn {
namespace ocl {

struct mutable_data_impl : public typed_primitive_impl_ocl<mutable_data> {
    using parent = typed_primitive_impl_ocl<mutable_data>;
    using parent::parent;
    static const object_type type;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<mutable_data_impl>(*this);
    }

    object_type get_type() const override {
        return type;
    }

    template <typename BufferType>
    void save(BufferType& buffer) const {}

    template <typename BufferType>
    void load(BufferType& buffer) {}

public:
    static primitive_impl* create(mutable_data_node const& arg) { return new mutable_data_impl(arg, {}); }
};

const object_type mutable_data_impl::type = object_type::MUTABLE_DATA_IMPL;

namespace detail {

attach_mutable_data_impl::attach_mutable_data_impl() {
    implementation_map<mutable_data>::add(impl_types::ocl, mutable_data_impl::create, {});
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::mutable_data_impl)
