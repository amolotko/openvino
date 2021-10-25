// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "permute_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "cldnn/runtime/error_handler.hpp"
#include "kernel_selector_helper.h"
#include "permute/permute_kernel_selector.h"
#include "permute/permute_kernel_ref.h"
#include "object_types.hpp"
#include "serialization/binary_buffer.hpp"

using namespace cldnn;

namespace cldnn {
namespace ocl {

struct permute_impl : typed_primitive_impl_ocl<permute> {
    using parent = typed_primitive_impl_ocl<permute>;
    using parent::parent;
    static const object_type type;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<permute_impl>(*this);
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

    static primitive_impl* create(const permute_node& arg) {
        auto permute_params = get_default_params<kernel_selector::permute_params>(arg);
        auto permute_optional_params =
            get_default_optional_params<kernel_selector::permute_optional_params>(arg.get_program());

        const auto& permute_order = arg.get_primitive()->permute_order;
        permute_params.order = permute_order;
        auto& kernel_selector = kernel_selector::permute_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(permute_params, permute_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        return new permute_impl(arg, best_kernels[0]);
    }
};

const object_type permute_impl::type = object_type::PERMUTE_IMPL;

namespace detail {

attach_permute_impl::attach_permute_impl() {
    implementation_map<permute>::add(impl_types::ocl, permute_impl::create, {});
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::permute_impl)
