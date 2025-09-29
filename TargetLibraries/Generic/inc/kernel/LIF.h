/* =====================================================================
 * Title:        LIF.h
 * Description:  Leaky Integrate-and-Fire neuron kernel (fp32, NCHW)
 *
 * Date:         25.09.2025
 *
 * ===================================================================== */
/*
 * Copyright (C) 2023 ETH Zurich and University of Bologna.
 *
 * Authors:
 * - Lorenzo Squarzoni, University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __DEEPLOY_BASIC_MATH_LIF_KERNEL_HEADER_
#define __DEEPLOY_BASIC_MATH_LIF_KERNEL_HEADER_

#include "DeeployBasicMath.h"

// LIF neuron update.
// input      : [N, C, H, W] input current
// mem_in     : [N, C, H, W] previous membrane
// beta       : [C]
// threshold  : [C]
// spike_out  : [N, C, H, W]
// mem_out    : [N, C, H, W]
void LIF_fp32_fp32(const float32_t *input, const float32_t *mem_in,
                   const float32_t *beta, const float32_t *threshold,
                   float32_t *spike_out, float32_t *mem_out, uint32_t N,
                   uint32_t C, uint32_t H, uint32_t W);

#endif // __DEEPLOY_BASIC_MATH_LIF_KERNEL_HEADER_
