// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "depth_to_space_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "kernel_selector_helper.h"
#include "depth_to_space/depth_to_space_kernel_selector.h"
#include "depth_to_space/depth_to_space_kernel_ref.h"
#include "cldnn/runtime/error_handler.hpp"
#include "common_types.h"
#include "object_types.hpp"
#include "serialization/binary_buffer.hpp"

using namespace cldnn;

namespace cldnn {
namespace ocl {
struct depth_to_space_impl : typed_primitive_impl_ocl<depth_to_space> {
    using parent = typed_primitive_impl_ocl<depth_to_space>;
    using parent::parent;
    static const object_type type;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<depth_to_space_impl>(*this);
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
    static primitive_impl* create(const depth_to_space_node& arg) {
        auto depth_to_space_params = get_default_params<kernel_selector::depth_to_space_params>(arg);
        auto depth_to_space_optional_params =
            get_default_optional_params<kernel_selector::depth_to_space_optional_params>(arg.get_program());

        depth_to_space_params.block_size = arg.get_primitive()->block_size;
        depth_to_space_params.mode = arg.get_primitive()->mode == depth_to_space_mode::blocks_first ? kernel_selector::depth_to_space_mode::BLOCKS_FIRST
                                                                                                    : kernel_selector::depth_to_space_mode::DEPTH_FIRST;

        auto& kernel_selector = kernel_selector::depth_to_space_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(depth_to_space_params, depth_to_space_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        return new depth_to_space_impl(arg, best_kernels[0]);
    }
};

const object_type depth_to_space_impl::type = object_type::DEPTH_TO_SPACE_IMPL;

namespace detail {

attach_depth_to_space_impl::attach_depth_to_space_impl() {
    implementation_map<depth_to_space>::add(impl_types::ocl, depth_to_space_impl::create, {
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::u8, format::bfyx),
        std::make_tuple(data_types::i8, format::bfyx),
        std::make_tuple(data_types::f32, format::bfzyx),
        std::make_tuple(data_types::f16, format::bfzyx),
        std::make_tuple(data_types::u8, format::bfzyx),
        std::make_tuple(data_types::i8, format::bfzyx),
        std::make_tuple(data_types::f32, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::f16, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::i8, format::b_fs_yx_fsv16),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::depth_to_space_impl)
