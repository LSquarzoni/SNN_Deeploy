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

// Stateful LIF neuron update (internal persistent membrane state).
// Semantics:
// - mem_state holds the persistent membrane across calls and is updated in-place.
// - If mem_in is non-NULL, it overrides the membrane just for this call; subsequent
//   calls continue from the updated mem_state.
// - On spike (updated_mem >= threshold[c]): spike_out=1, mem_state reset to 0.
// - Otherwise: spike_out=0, mem_state=updated_mem.
// input      : [N, C, H, W] input current
// mem_in     : optional [N, C, H, W] override for this call (pass NULL to use mem_state)
// beta       : [C]          decay factor per channel (0..1)
// threshold  : [C]          firing threshold per channel
// spike_out  : [N, C, H, W] output spikes (0/1)
// mem_state  : [N, C, H, W] in/out persistent membrane state buffer
// N, C, H, W : tensor dimensions
void LIF_stateful_fp32(const float32_t *input, const float32_t *mem_in,
                       const float32_t *beta, const float32_t *threshold,
                       float32_t *spike_out, float32_t *mem_state,
                       uint32_t N, uint32_t C, uint32_t H, uint32_t W);

// LIF neuron update with int8 quantization.
// Activations (input, mem, spike, mem_out) are int8.
// Parameters (beta, threshold) remain fp32 as exported from ONNX.
// input      : [N, C, H, W] input current (int8)
// mem_in     : [N, C, H, W] previous membrane (int8)
// beta       : [C]          decay factor per channel (fp32, from ONNX)
// threshold  : [C]          firing threshold per channel (fp32, from ONNX)
// spike_out  : [N, C, H, W] output spikes (int8)
// mem_out    : [N, C, H, W] updated membrane (int8)
// input_scale, mem_scale    : scale factors for dequantization
void LIF_s8_s8_s32(const int8_t *input, const int8_t *mem_in,
                   const float32_t *beta, const float32_t *threshold,
                   int8_t *spike_out, int8_t *mem_out, int32_t input_offset,
                   int32_t mem_in_offset, int32_t output_offset,
                   int32_t mem_out_offset, float32_t input_scale,
                   float32_t mem_scale, uint32_t N, uint32_t C, uint32_t H,
                   uint32_t W);

#endif // __DEEPLOY_BASIC_MATH_LIF_KERNEL_HEADER_
