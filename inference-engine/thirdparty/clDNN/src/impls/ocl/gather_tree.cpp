// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_tree_inst.h"

#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "kernel_selector_helper.h"
#include "gather_tree/gather_tree_kernel_selector.h"
#include "gather_tree/gather_tree_kernel_base.h"
#include "cldnn/runtime/error_handler.hpp"
#include "object_types.hpp"
#include "serialization/binary_buffer.hpp"

namespace cldnn {
namespace ocl {

struct gather_tree_impl : typed_primitive_impl_ocl<gather_tree> {
    using parent = typed_primitive_impl_ocl<gather_tree>;
    using parent::parent;
    static const object_type type;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<gather_tree_impl>(*this);
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

    static primitive_impl* create(const gather_tree_node& arg) {
        auto b_params = get_default_params<kernel_selector::gather_tree_params>(arg, 1);
        auto b_optional_params = get_default_optional_params<kernel_selector::gather_tree_optional_params>(arg.get_program());

        for (size_t i = 1; i < arg.get_dependencies().size(); i++) {
            b_params.inputs.push_back(convert_data_tensor(arg.get_dependency(i).get_output_layout(), 1));
        }
        auto desc = arg.get_primitive();

        auto& kernel_selector = kernel_selector::gather_tree_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(b_params, b_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
            "Best_kernel.empty()",
            best_kernels.empty(),
            "Cannot find a proper kernel with this arguments");

        return new gather_tree_impl(arg, best_kernels[0]);
    }
};

const object_type gather_tree_impl::type = object_type::GATHER_TREE_IMPL;

namespace detail {
attach_gather_tree_impl::attach_gather_tree_impl() {
    implementation_map<gather_tree>::add(impl_types::ocl, gather_tree_impl::create, {
        std::make_tuple(data_types::i32, format::yxfb),
        std::make_tuple(data_types::i32, format::bfyx),
        std::make_tuple(data_types::i32, format::byxf),
        std::make_tuple(data_types::f32, format::yxfb),
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f32, format::byxf),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::gather_tree_impl)
