// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_elements_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "kernel_selector_helper.h"
#include "gather/gather_elements_kernel_selector.h"
#include "gather/gather_elements_kernel_ref.h"
#include "cldnn/runtime/error_handler.hpp"
#include "object_types.hpp"
#include "serialization/binary_buffer.hpp"

using namespace cldnn;

namespace cldnn {
namespace ocl {
kernel_selector::gather_elements_axis convert_axis(gather_elements::gather_elements_axis axis) {
    switch (axis) {
        case gather_elements::along_x:
            return kernel_selector::gather_elements_axis::X;
        case gather_elements::along_y:
            return kernel_selector::gather_elements_axis::Y;
        case gather_elements::along_z:
            return kernel_selector::gather_elements_axis::Z;
        case gather_elements::along_w:
            return kernel_selector::gather_elements_axis::W;
        case gather_elements::along_f:
            return kernel_selector::gather_elements_axis::FEATURE;
        case gather_elements::along_b:
            return kernel_selector::gather_elements_axis::BATCH;
        default:
            return kernel_selector::gather_elements_axis::BATCH;
    }
}

struct gather_elements_impl : typed_primitive_impl_ocl<gather_elements> {
    using parent = typed_primitive_impl_ocl<gather_elements>;
    using parent::parent;
    static const object_type type;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<gather_elements_impl>(*this);
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

public:
    static primitive_impl* create(const gather_elements_node& arg) {
        auto gather_elements_params = get_default_params<kernel_selector::gather_elements_params>(arg);
        auto gather_elements_optional_params =
            get_default_optional_params<kernel_selector::gather_elements_optional_params>(arg.get_program());

        gather_elements_params.axis = convert_axis(arg.get_primitive()->axis);

        gather_elements_params.inputs.push_back(convert_data_tensor(arg.input(1).get_output_layout()));

        auto& kernel_selector = kernel_selector::gather_elements_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(gather_elements_params, gather_elements_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        return new gather_elements_impl(arg, best_kernels[0]);
    }
};

const object_type gather_elements_impl::type = object_type::GATHER_ELEMENTS_IMPL;

namespace detail {

attach_gather_elements_impl::attach_gather_elements_impl() {
    implementation_map<gather_elements>::add(impl_types::ocl, gather_elements_impl::create, {
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

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::gather_elements_impl)
