// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/strided_slice.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

std::vector<StridedSliceSpecificParams> ss_only_test_cases = {
        StridedSliceSpecificParams{ { 16 }, { 4 }, { 12 }, { 1 },
                                    { 0 }, { 0 },  { },  { },  { } },
        StridedSliceSpecificParams{ { 16 }, { 0 }, { 8 }, { 2 },
                                    { 1 }, { 0 },  { },  { },  { } },
        StridedSliceSpecificParams{ { 128, 1 }, { 0, 0, 0 }, { 0, 0, 0 }, { 1, 1, 1 },
                            { 0, 1, 1 }, { 0, 1, 1 },  { 1, 0, 0 },  { 0, 0, 0 },  { 0, 0, 0 } },
        StridedSliceSpecificParams{ { 128, 1 }, { 0, 0, 0 }, { 0, 0, 0 }, { 1, 1, 1},
                            { 1, 0, 1 }, { 1, 0, 1 },  { 0, 1, 0 },  { 0, 0, 0 },  { 0, 0, 0 } },
        StridedSliceSpecificParams{ { 2, 3 }, { 1, 0 }, { 2, 3 }, { 1, 1 },
                            { 0, 0 }, { 0, 0 },  {  },  {  },  {  } },
        StridedSliceSpecificParams{ { 10, 3 }, { 0, 0 }, { 20, 20 }, { 1, 1 },
                            { 0, 1 }, { 0, 1 },  {  },  {  },  {  } },
        StridedSliceSpecificParams{ { 1, 12, 100 }, { 0, -1, 0 }, { 0, 0, 0 }, { 1, 1, 1 },
                            { 1, 0, 1 }, { 1, 0, 1 },  { 0, 0, 0 },  { 0, 1, 0 },  { 0, 0, 0 } },
        StridedSliceSpecificParams{ { 1, 12, 100 }, { 0, 9, 0 }, { 0, 11, 0 }, { 1, 1, 1 },
                            { 1, 0, 1 }, { 1, 0, 1 },  { 0, 0, 0 },  { 0, 0, 0 },  { 0, 0, 0 } },
        StridedSliceSpecificParams{ { 1, 12, 100 }, { 0, 1, 0 }, { 0, -1, 0 }, { 1, 1, 1 },
                            { 1, 0, 1 }, { 1, 0, 1 },  { 0, 0, 0 },  { 0, 0, 0 },  { 0, 0, 0 } },
        StridedSliceSpecificParams{ { 2, 12, 100 }, { 0, 9, 0 }, { 0, 7, 0 }, { -1, -1, -1 },
                            { 1, 0, 1 }, { 1, 0, 1 },  { 0, 0, 0 },  { 0, 0, 0 },  { 0, 0, 0 } },
        StridedSliceSpecificParams{ { 2, 12, 100 }, { 0, 7, 0 }, { 0, 9, 0 }, { -1, 1, -1 },
                            { 1, 0, 1 }, { 1, 0, 1 },  { 0, 0, 0 },  { 0, 0, 0 },  { 0, 0, 0 } },
        StridedSliceSpecificParams{ { 1, 12, 100 }, { 0, 4, 0 }, { 0, 9, 0 }, { -1, 2, -1 },
                            { 1, 0, 1 }, { 1, 0, 1 },  { 0, 0, 0 },  { 0, 0, 0 },  { 0, 0, 0 } },
        StridedSliceSpecificParams{ { 1, 12, 100 }, { 0, 4, 0 }, { 0, 10, 0 }, { -1, 2, -1 },
                            { 1, 0, 1 }, { 1, 0, 1 },  { 0, 0, 0 },  { 0, 0, 0 },  { 0, 0, 0 } },
        StridedSliceSpecificParams{ { 1, 12, 100 }, { 0, 9, 0 }, { 0, 4, 0 }, { -1, -2, -1 },
                            { 1, 0, 1 }, { 1, 0, 1 },  { 0, 0, 0 },  { 0, 0, 0 },  { 0, 0, 0 } },
        StridedSliceSpecificParams{ { 2, 12, 100 }, { 0, 10, 0 }, { 0, 4, 0 }, { -1, -2, -1 },
                            { 1, 0, 1 }, { 1, 0, 1 },  { 0, 0, 0 },  { 0, 0, 0 },  { 0, 0, 0 } },
        StridedSliceSpecificParams{ { 1, 12, 100 }, { 0, 11, 0 }, { 0, 0, 0 }, { -1, -2, -1 },
                            { 1, 0, 1 }, { 1, 0, 1 },  { 0, 0, 0 },  { 0, 0, 0 },  { 0, 0, 0 } },
        StridedSliceSpecificParams{ { 1, 12, 100 }, { 0, -6, 0 }, { 0, -8, 0 }, { -1, -2, -1 },
                            { 1, 0, 1 }, { 1, 0, 1 },  { 0, 0, 0 },  { 0, 0, 0 },  { 0, 0, 0 } },
        StridedSliceSpecificParams{ { 20, 10, 5 }, { 0, 0, 0 }, { 3, 10, 0 }, { 1, 1, 1 },
                            { 0, 0, 1 }, { 0, 0, 1 },  { 0, 0, 0 },  { 0, 0, 0 },  { 0, 0, 0 } },
        StridedSliceSpecificParams{ { 1, 10, 20 }, { 0, 0, 2 }, { 0, 0, 1000 }, { 1, 1, 1 },
                            { 1, 1, 0 }, { 1, 1, 0 },  {  },  {  },  {  } },
        StridedSliceSpecificParams{ { 1, 10, 10 }, { 0, 1, 0 }, { 0, 1000, 0 }, { 1, 1, 1 },
                            { 1, 0, 1 }, { 1, 0, 1 },  {  },  {  },  {  } },
        StridedSliceSpecificParams{ { 1, 10, 4 }, { 0, 0, 0 }, { 0, 0, 2 }, { 1, 1, 1 },
                            { 1, 1, 0 }, { 1, 1, 0 },  {  },  {  },  {  } },
        StridedSliceSpecificParams{ { 1, 10, 4 }, { 0, 0, 2 }, { 0, 0, 1000 }, { 1, 1, 1 },
                            { 1, 1, 0 }, { 1, 1, 0 },  {  },  {  },  {  } },
        StridedSliceSpecificParams{ { 1, 10, 2 }, { 0, 0, 0 }, { 0, 0, 1 }, { 1, 1, 1 },
                            { 1, 1, 0 }, { 1, 1, 0 },  {  },  {  },  {  } },
        StridedSliceSpecificParams{ { 1, 10, 2 }, { 0, 0, 0 }, { 1000, 0, 0 }, { 1, 1, 1 },
                            { 0, 1, 1 }, { 0, 1, 1 },  {  },  {  },  {  } },
        StridedSliceSpecificParams{ { 1, 10, 2 }, { 0, 0, 0 }, { 0, 1000, 0 }, { 1, 1, 1 },
                            { 1, 0, 1 }, { 1, 0, 1 },  {  },  {  },  {  } },
        StridedSliceSpecificParams{ { 20, 10, 5 }, { 0, 3 }, { 0, 4 }, { 1, 1 },
                            { 1, 0 }, { 1, 0 },  {  },  {  },  { 1, 0 } },
        StridedSliceSpecificParams{ { 20, 10, 5 }, { 0, 0 }, { 0, -1 }, { 1, 1 },
                            { 1, 0 }, { 1, 0 },  {  },  {  },  { 1, 0 } },
        StridedSliceSpecificParams{ { 20, 10, 5 }, { 0, 0 }, { 0, -1 }, { 1, 1 },
                                    { 1, 0 }, { 1, 0 },  { 0, 0 },  { 0, 0 },  { 0, 0 } },
        StridedSliceSpecificParams{ { 1, 12, 100, 1, 1 }, { 0, -1, 0, 0 }, { 0, 0, 0, 0 }, { 1, 1, 1, 1 },
                            { 1, 0, 1, 0 }, { 1, 0, 1, 0 },  { },  { 0, 1, 0, 1 },  {} },
        StridedSliceSpecificParams{ { 2, 2, 2, 2 }, { 0, 0, 0, 0 }, { 2, 2, 2, 2 }, { 1, 1, 1, 1 },
                            { 1, 1, 1, 1}, { 1, 1, 1, 1},  {},  {},  {} },
        StridedSliceSpecificParams{ { 2, 2, 2, 2 }, { 0, 0 }, { 2, 2 }, { 1, 1 },
                            { 1, 1 }, { 1, 1 },  {},  {},  {} },
        StridedSliceSpecificParams{ { 2, 2, 3, 3 }, { 0, -2, -2 }, { 2, -1, -1 }, { 1, 1, 1 },
                            { 1, 0 }, { 1, 0 },  {},  {},  {} },
        StridedSliceSpecificParams{ { 2, 2, 2, 2 }, { 1, 1, 1, 1 }, { 2, 2, 2, 2 }, { 1, 1, 1, 1 },
                            { 0, 0, 0, 0}, { 1, 1, 1, 1},  {},  {},  {} },
        StridedSliceSpecificParams{ { 2, 2, 2, 2 }, { 1, 1, 1, 1 }, { 2, 2, 2, 2 }, { 1, 1, 1, 1 },
                            { 0, 0, 0, 0}, { 0, 0, 0, 0},  {},  {},  {} },
        StridedSliceSpecificParams{ { 1, 2, 6, 4 }, { 0, 0, 4, 0 }, { 1, 2, 6, 4 }, { 1, 1, 1, 1 },
                            { 0, 0, 0, 0 }, { 0, 0, 0, 0 }, {}, {}, {} },
        StridedSliceSpecificParams{ { 1, 2, 6, 4 }, { 0, 0, -3, 0 }, { 1, 2, 6, 4 }, { 1, 1, 1, 1 },
                            { 0, 0, 0, 0 }, { 0, 0, 0, 0 }, {}, {}, {} },
        StridedSliceSpecificParams{ { 1, 2, 6, 4 }, { 0, 0, 4, 0 }, { 1, 2, 6, 4 }, { 1, 1, 1, 1 },
                            { 1, 1, 0, 1}, { 1, 1, 1, 1},  {},  {},  {} },
        StridedSliceSpecificParams{ { 10, 2, 2, 2 }, { 0, 0, 0, 0 }, { 0, 0, 0, 0 }, { 2, 1, 1, 1 },
                            { 1, 1, 1, 1}, { 1, 1, 1, 1},  {},  {},  {} },
        StridedSliceSpecificParams{ { 2, 2, 4, 3 }, { 0, 0, 0, 0 }, { 2, 2, 4, 3 }, { 1, 1, 2, 1 },
                            { 1, 1, 1, 1}, { 1, 1, 1, 1},  {},  {},  {} },
        StridedSliceSpecificParams{ { 2, 2, 4, 2 }, { 1, 0, 0, 1 }, { 2, 2, 4, 2 }, { 1, 1, 2, 1 },
                            { 0, 1, 1, 0}, { 1, 1, 0, 0},  {},  {},  {} },
        StridedSliceSpecificParams{ { 1, 2, 4, 2 }, { 1, 0, 0, 0 }, { 1, 2, 4, 2 }, { 1, 1, -2, -1 },
                            { 1, 1, 1, 1}, { 1, 1, 1, 1},  {},  {},  {} },
        StridedSliceSpecificParams{ { 2, 2, 4, 2 }, { 1, 0, 0, 0 }, { 1, 2, 4, 2 }, { 1, 1, -2, -1 },
                            { 0, 1, 1, 1}, { 1, 1, 1, 1},  {},  {},  {} },
        StridedSliceSpecificParams{ { 2, 3, 4, 5, 6 }, { 0, 1, 0, 0, 0 }, { 2, 3, 4, 5, 6 }, { 1, 1, 1, 1, 1 },
                            { 1, 0, 1, 1, 1}, { 1, 0, 1, 1, 1 },  {},  { 0, 1, 0, 0, 0 },  {} },
        StridedSliceSpecificParams{ { 2, 3, 4, 5, 6 }, { 0, 0, 3, 0, 0 }, { 2, 3, 4, 3, 6 }, { 1, 1, 1, 1, 1 },
                            { 1, 1, 0, 1, 1}, { 1, 1, 0, 0, 1 },  {},  { 0, 0, 1, 0, 0 },  {} },
        StridedSliceSpecificParams{ { 2, 3, 4, 5, 6 }, { 0, 0, 0, 0, 3 }, { 1, 3, 4, 5, 6 }, { 1, 1, 1, 1, 1 },
                            { 0, 1, 1, 1, 0}, { 0, 1, 1, 1, 0 },  {},  { 1, 0, 0, 0, 1 },  {} },
        StridedSliceSpecificParams{ { 2, 3, 4, 5 }, { 0, 0, 0, 0, 0 }, { 0, 2, 3, 4, 5 }, { 1, 1, 1, 1, 1 },
                            { 1, 1, 1, 1, 1 }, { 1, 1, 1, 1, 1 },  { 1, 0, 0, 0, 0 },  {},  {} },
        StridedSliceSpecificParams{ { 2, 3, 4, 5 }, { 0, 0, 0, 0, 0 }, { 0, 2, 3, 4, 5 }, { 1, 1, 1, 1, 1 },
                            { 1, 1, 1, 1, 1 }, { 1, 1, 1, 1, 1 },  { 0, 0, 1, 0, 0 },  {},  {} },
        StridedSliceSpecificParams{ { 10, 12 }, { -1, 1 }, { -9999, 0 }, { -1, 1 },
                            { 0, 1 }, { 0, 1 },  { 0, 0 },  { 0, 0 },  { 0, 0 } },
        StridedSliceSpecificParams{ { 5, 5, 5, 5 }, { -1, 0, -1, 0 }, { -50, 0, -60, 0 }, { -1, 1, -1, 1 },
                            { 0, 0, 0, 0 }, { 0, 1, 0, 1 },  { 0, 0, 0, 0 },  { 0, 0, 0, 0 },  { 0, 0, 0, 0 } },
        StridedSliceSpecificParams{ { 1, 2, 4 }, { 0, 2000, 3, 5 }, { 0, 0, 0, 2 }, { 1, 1, 1, 1 },
                            { 1, 0, 1, 1 }, { 1, 0, 1, 0 },  { 0, 1, 0, 0 },  { },  { } },
        StridedSliceSpecificParams{ { 2, 2, 4, 4 }, { 0, 0, 0, 1, 0 }, { 0, 0, 0, 2, 0 }, { 1, 1, 1, 1, 1 },
                            { 1, 1, 1, 0, 1 }, { 1, 1, 1, 0, 1 },  { 0, 1, 0, 0, 0 },  { },  { } },
        StridedSliceSpecificParams{ { 2, 2, 2, 4, 4 }, { 0, 0, 0, 1, 0 }, { 0, 0, 0, 2, 0 }, { 1, 1, 1, 1, 1 },
                            { 1, 1, 1, 0, 1 }, { 1, 1, 1, 0, 1 },  { },  { 0, 1, 0, 0, 0 },  { } },
};

INSTANTIATE_TEST_SUITE_P(
        smoke_INTEL_CPU, StridedSliceLayerTest,
        ::testing::Combine(
            ::testing::ValuesIn(ss_only_test_cases),
            ::testing::Values(InferenceEngine::Precision::FP32),
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(CommonTestUtils::DEVICE_CPU),
            ::testing::Values(std::map<std::string, std::string>())),
        StridedSliceLayerTest::getTestCaseName);

}  // namespace
