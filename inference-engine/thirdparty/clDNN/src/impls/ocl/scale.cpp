// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "scale_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "kernel_selector_helper.h"
#include "eltwise/eltwise_kernel_selector.h"
#include "eltwise/eltwise_kernel_base.h"
#include "cldnn/runtime/error_handler.hpp"
#include "object_types.hpp"
#include "serialization/binary_buffer.hpp"

using namespace cldnn;

namespace cldnn {
namespace ocl {

struct scale_impl : typed_primitive_impl_ocl<scale> {
    using parent = typed_primitive_impl_ocl<scale>;
    using parent::parent;
    static const object_type type;

    scale_impl(const scale_impl& other) : parent(other), _bias_term(other._bias_term) {}

    scale_impl(const scale_node& arg, const kernel_selector::kernel_data& kd) : parent(arg, kd), _bias_term(arg.bias_term()) {}

    void align_state(const program_node& arg) override {
        if (!arg.is_type<scale>()) {
            throw std::invalid_argument("Should be scale node");
        }
        const auto& scale_node = arg.as<scale>();
        _bias_term = scale_node.bias_term();
    }

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<scale_impl>(*this);
    }

    object_type get_type() const override {
        return type;
    }

    template <typename BufferType>
    void save(BufferType& buffer) const {}

    template <typename BufferType>
    void load(BufferType& buffer) {}

protected:
    kernel_arguments_data get_arguments(typed_primitive_inst<scale>& instance, int32_t split) const override {
        kernel_arguments_data args = parent::get_arguments(instance, split);
        args.inputs = {instance.input_memory_ptr(), instance.scale_memory()};
        args.output = instance.output_memory_ptr();

        if (_bias_term) {
            args.inputs.push_back(instance.bias_memory());
        }
        return args;
    }

public:
    static primitive_impl* create(const scale_node& arg) {
        auto ew_params = get_default_params<kernel_selector::eltwise_params>(arg);
        auto ew_optional_params =
            get_default_optional_params<kernel_selector::eltwise_optional_params>(arg.get_program());

        ew_params.inputs.push_back(convert_data_tensor(arg.scale_in().get_output_layout()));

        ew_params.operations.push_back({{kernel_selector::eltwise_params::InputType::Buffer(0),
                                         kernel_selector::eltwise_params::InputType::Buffer(1)},
                                        kernel_selector::eltwise_mode::MUL});

        if (arg.bias_term()) {
            ew_params.inputs.push_back(convert_data_tensor(arg.bias().get_output_layout()));
            ew_params.operations.push_back({{kernel_selector::eltwise_params::InputType::Intermediate(0),
                                             kernel_selector::eltwise_params::InputType::Buffer(2)},
                                            kernel_selector::eltwise_mode::ADD});
        }

        ew_params.layoutBased = true;

        auto& kernel_selector = kernel_selector::eltwise_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(ew_params, ew_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        return new scale_impl(arg, best_kernels[0]);
    }

private:
    bool _bias_term = false;
};

const object_type scale_impl::type = object_type::SCALE_IMPL;

namespace detail {

attach_scale_impl::attach_scale_impl() {
    implementation_map<scale>::add(impl_types::ocl, scale_impl::create, {
        std::make_tuple(data_types::f32, format::yxfb),
        std::make_tuple(data_types::f16, format::yxfb),
        std::make_tuple(data_types::i32, format::yxfb),
        std::make_tuple(data_types::f32, format::byxf),
        std::make_tuple(data_types::f16, format::byxf),
        std::make_tuple(data_types::i32, format::byxf),
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::i32, format::bfyx),
        std::make_tuple(data_types::u8, format::bfyx),
        std::make_tuple(data_types::i8, format::bfyx),
        std::make_tuple(data_types::f32, format::bfzyx),
        std::make_tuple(data_types::f16, format::bfzyx),
        std::make_tuple(data_types::i32, format::bfzyx),
        std::make_tuple(data_types::u8, format::bfzyx),
        std::make_tuple(data_types::i8, format::bfzyx),
        std::make_tuple(data_types::f32, format::bfwzyx),
        std::make_tuple(data_types::f16, format::bfwzyx),
        std::make_tuple(data_types::i32, format::bfwzyx),
        std::make_tuple(data_types::u8, format::bfwzyx),
        std::make_tuple(data_types::i8, format::bfwzyx),
        std::make_tuple(data_types::f32, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::f16, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::i32, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::i8, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::f32, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::f16, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::i32, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::i8, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::u8, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::f32, format::bs_fs_zyx_bsv16_fsv16),
        std::make_tuple(data_types::f16, format::bs_fs_zyx_bsv16_fsv16),
        std::make_tuple(data_types::i32, format::bs_fs_zyx_bsv16_fsv16),
        std::make_tuple(data_types::f16, format::fs_b_yx_fsv32),
        std::make_tuple(data_types::i32, format::fs_b_yx_fsv32),
        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv16_fsv16),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv16_fsv16),
        std::make_tuple(data_types::i32, format::bs_fs_yx_bsv16_fsv16),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv4),
        std::make_tuple(data_types::i8, format::b_fs_yx_fsv4),
        std::make_tuple(data_types::f16, format::b_fs_yx_fsv4),
        std::make_tuple(data_types::f32, format::b_fs_yx_fsv4),
        std::make_tuple(data_types::i32, format::b_fs_yx_fsv4),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::i8, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::f16, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::f32, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::i32, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::u8, format::b_fs_zyx_fsv32),
        std::make_tuple(data_types::i8, format::b_fs_zyx_fsv32),
        std::make_tuple(data_types::f16, format::b_fs_zyx_fsv32),
        std::make_tuple(data_types::f32, format::b_fs_zyx_fsv32),
        std::make_tuple(data_types::i32, format::b_fs_zyx_fsv32),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::scale_impl)
