// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_nd_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "kernel_selector_helper.h"
#include "gather/gather_nd_kernel_selector.h"
#include "gather/gather_nd_kernel_ref.h"
#include "object_types.hpp"
#include "serialization/binary_buffer.hpp"

using namespace cldnn;

namespace cldnn {
namespace ocl {

struct gather_nd_impl : typed_primitive_impl_ocl<gather_nd> {
    using parent = typed_primitive_impl_ocl<gather_nd>;
    using parent::parent;
    static const object_type type;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<gather_nd_impl>(*this);
    }

    object_type get_type() const override {
        return type;
    }

    template <typename BufferType>
    void save(BufferType& buffer) const {
        parent::save(buffer);
    }

    template <typename BufferType>
    void load(BufferType& buffer) {
        parent::load(buffer);
    }

    static primitive_impl* create(const gather_nd_node& arg) {
        auto gather_nd_params = get_default_params<kernel_selector::gather_nd_params>(arg);
        auto gather_nd_optional_params =
            get_default_optional_params<kernel_selector::gather_nd_optional_params>(arg.get_program());

        gather_nd_params.indices_rank = arg.get_primitive()->indices_rank;
        gather_nd_params.batch_dims = arg.get_primitive()->batch_dims;

        gather_nd_params.inputs.push_back(convert_data_tensor(arg.input(1).get_output_layout()));

        auto& kernel_selector = kernel_selector::gather_nd_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(gather_nd_params, gather_nd_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        return new gather_nd_impl(arg, best_kernels[0]);
    }
};

const object_type gather_nd_impl::type = object_type::GATHER_ND_IMPL;

namespace detail {

attach_gather_nd_impl::attach_gather_nd_impl() {
    implementation_map<gather_nd>::add(impl_types::ocl, gather_nd_impl::create, {
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::i32, format::bfyx),
        std::make_tuple(data_types::f32, format::bfzyx),
        std::make_tuple(data_types::f16, format::bfzyx),
        std::make_tuple(data_types::i32, format::bfzyx),
        std::make_tuple(data_types::f32, format::bfwzyx),
        std::make_tuple(data_types::f16, format::bfwzyx),
        std::make_tuple(data_types::i32, format::bfwzyx),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::gather_nd_impl)
