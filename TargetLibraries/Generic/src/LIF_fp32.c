/* =====================================================================
 * Title:        LIF_fp32.c
 * Description:  Leaky Integrate-and-Fire neuron update (fp32, NCHW)
 *
 * Date:         25.09.2025
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
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "DeeployBasicMath.h"

// LIF neuron update.
// input      : [N, C, H, W] input current
// mem_in     : [N, C, H, W] previous membrane potential
// beta       : [C]          decay factor per channel (0..1)
// threshold  : [C]          firing threshold per channel
// spike_out  : [N, C, H, W] output spikes (0/1)
// mem_out    : [N, C, H, W] updated membrane potential (reset to 0 on spike)
// N, C, H, W : tensor dimensions
// Reset mechanism brings the membrane potential to zero upon spiking

void LIF_fp32_fp32(const float32_t *__restrict__ input,
									 const float32_t *__restrict__ mem_in,
									 const float32_t *__restrict__ beta,
									 const float32_t *__restrict__ threshold,
									 float32_t *__restrict__ spike_out,
									 float32_t *__restrict__ mem_out, uint32_t N, uint32_t C,
									 uint32_t H, uint32_t W) {

	for (uint32_t n = 0; n < N; ++n) {
		for (uint32_t c = 0; c < C; ++c) {
			float32_t beta_val = beta[c];
			float32_t thr_val = threshold[c];
			for (uint32_t h = 0; h < H; ++h) {
				for (uint32_t w = 0; w < W; ++w) {
					uint32_t idx = ((n * C + c) * H + h) * W + w;
					float32_t updated_mem = beta_val * mem_in[idx] + input[idx];
					if (updated_mem >= thr_val) {
						spike_out[idx] = 1.0f;
						mem_out[idx] = 0.0f; // reset after spike
					} else {
						spike_out[idx] = 0.0f;
						mem_out[idx] = updated_mem;
					}
				}
			}
		}
	}
}
