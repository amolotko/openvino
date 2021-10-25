// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "region_yolo_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "kernel_selector_helper.h"
#include "region_yolo/region_yolo_kernel_selector.h"
#include "region_yolo/region_yolo_kernel_ref.h"
#include "cldnn/runtime/error_handler.hpp"
#include "object_types.hpp"
#include "serialization/binary_buffer.hpp"

namespace cldnn {
namespace ocl {

struct region_yolo_impl : typed_primitive_impl_ocl<region_yolo> {
    using parent = typed_primitive_impl_ocl<region_yolo>;
    using parent::parent;
    static const object_type type;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<region_yolo_impl>(*this);
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

    static primitive_impl* create(const region_yolo_node& arg) {
        auto ry_params = get_default_params<kernel_selector::region_yolo_params>(arg);
        auto ry_optional_params =
            get_default_optional_params<kernel_selector::region_yolo_optional_params>(arg.get_program());

        const auto& primitive = arg.get_primitive();
        ry_params.coords = primitive->coords;
        ry_params.classes = primitive->classes;
        ry_params.num = primitive->num;
        ry_params.do_softmax = primitive->do_softmax;
        ry_params.mask_size = primitive->mask_size;

        auto& kernel_selector = kernel_selector::region_yolo_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(ry_params, ry_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        return new region_yolo_impl(arg, best_kernels[0]);
    }
};

const object_type region_yolo_impl::type = object_type::REGION_YOLO_IMPL;

namespace detail {

attach_region_yolo_impl::attach_region_yolo_impl() {
    implementation_map<region_yolo>::add(impl_types::ocl, region_yolo_impl::create, {
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::f32, format::byxf),
        std::make_tuple(data_types::f16, format::byxf),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::region_yolo_impl)
