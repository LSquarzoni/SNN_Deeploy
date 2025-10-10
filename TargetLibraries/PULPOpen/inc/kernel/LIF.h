/* =====================================================================
 * Title:        LIF.h
 * Description:  Leaky Integrate-and-Fire neuron kernel (fp32, NCHW)
 *
 * Date:         02.10.2025
 *
 * ===================================================================== */
/*
 * Copyright (C) 2025 ETH Zurich and University of Bologna.
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
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __DEEPLOY_PULP_MATH_LIF_KERNEL_HEADER_
#define __DEEPLOY_PULP_MATH_LIF_KERNEL_HEADER_

#include "DeeployPULPMath.h"

// PULP-optimized LIF neuron update (fp32, NHWC), parallelized over channels.
// input      : [N, H, W, C] input current
// mem_in     : [N, H, W, C] previous membrane
// beta       : [C]          decay factor per channel (0..1)
// threshold  : [C]          firing threshold per channel
// spike_out  : [N, H, W, C] output spikes (0/1)
// mem_out    : [N, H, W, C] updated membrane potential (reset to 0 on spike)
void PULP_LIF_fp32_fp32(const float32_t *input, const float32_t *mem_in,
                        const float32_t *beta, const float32_t *threshold,
                        float32_t *spike_out, float32_t *mem_out, uint32_t N,
                        uint32_t H, uint32_t W, uint32_t C);

// PULP-optimized stateful LIF neuron update (fp32, NHWC); updates mem_state in-place and can
// optionally take a mem_in for this invocation. If mem_in is NULL, mem_state is used as prev mem.
void PULP_LIF_stateful_fp32_fp32(const float32_t *input, const float32_t *mem_in,
                                 const float32_t *beta, const float32_t *threshold,
                                 float32_t *spike_out, float32_t *mem_state, uint32_t N,
                                 uint32_t H, uint32_t W, uint32_t C);

#endif // __DEEPLOY_PULP_MATH_LIF_KERNEL_HEADER_
