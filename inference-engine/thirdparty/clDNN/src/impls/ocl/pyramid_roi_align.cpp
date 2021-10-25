// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "kernel_selector_helper.h"
#include "pyramid_roi_align/pyramid_roi_align_kernel_selector.h"
#include "pyramid_roi_align/pyramid_roi_align_kernel_base.h"
#include "cldnn/runtime/error_handler.hpp"
#include "pyramid_roi_align_inst.h"
#include "object_types.hpp"
#include "serialization/binary_buffer.hpp"

#include <cmath>

namespace cldnn {
namespace ocl {

struct pyramid_roi_align_impl : typed_primitive_impl_ocl<pyramid_roi_align> {
    using parent = typed_primitive_impl_ocl<pyramid_roi_align>;
    using parent::parent;
    static const object_type type;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<pyramid_roi_align_impl>(*this);
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

    static primitive_impl* create(const pyramid_roi_align_node& arg) {
        auto prim = arg.get_primitive();
        auto params = get_default_params<kernel_selector::PyramidROIAlign_params>(arg, 1);
        auto optional_params =
            get_default_optional_params<kernel_selector::PyramidROIAlign_optional_params>(arg.get_program());

        params.inputs.push_back(convert_data_tensor(arg.P2().get_output_layout()));
        params.inputs.push_back(convert_data_tensor(arg.P3().get_output_layout()));
        params.inputs.push_back(convert_data_tensor(arg.P4().get_output_layout()));
        params.inputs.push_back(convert_data_tensor(arg.P5().get_output_layout()));

        params.sampling_ratio_x = prim->sampling_ratio;
        params.sampling_ratio_y = prim->sampling_ratio;

        auto first_layer_scale = prim->pyramid_scales[0];
        auto image_size_x = arg.P2().get_output_layout().size.spatial[0] * first_layer_scale;
        auto image_size_y = arg.P2().get_output_layout().size.spatial[1] * first_layer_scale;
        params.image_size_x = image_size_x;
        params.image_size_y = image_size_y;

        params.pyramid_starting_level = prim->pyramid_starting_level;

        auto& kernel_selector = kernel_selector::PyramidROIAlign_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(params, optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        return new pyramid_roi_align_impl(arg, best_kernels[0]);
    }
};

const object_type pyramid_roi_align_impl::type = object_type::PYRAMID_ROI_ALIGN_IMPL;

namespace detail {

attach_pyramid_roi_align_impl::attach_pyramid_roi_align_impl() {
    implementation_map<pyramid_roi_align>::add(impl_types::ocl, pyramid_roi_align_impl::create, {
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f32, format::yxfb),
        std::make_tuple(data_types::f32, format::byxf),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::f16, format::yxfb),
        std::make_tuple(data_types::f16, format::byxf),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::pyramid_roi_align_impl)
