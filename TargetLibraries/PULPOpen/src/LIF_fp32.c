/* =====================================================================
 * Title:        LIF_fp32.c
 * Description:  Leaky Integrate-and-Fire neuron update (fp32, NCHW) for PULP
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

#include "DeeployPULPMath.h"
#include "pmsis.h"

// Parallelization strategy: split channels C across cores. Each core processes
// a contiguous chunk of channels for all batches and spatial positions.
// Tensor layout is NHWC: index = ((n * H + h) * W + w) * C + c
void PULP_LIF_fp32_fp32(const float32_t *__restrict__ input,
												const float32_t *__restrict__ mem_in,
												const float32_t *__restrict__ beta,
												const float32_t *__restrict__ threshold,
												float32_t *__restrict__ spike_out,
												float32_t *__restrict__ mem_out, uint32_t N,
												uint32_t C, uint32_t H, uint32_t W) {

	int8_t core_id = pi_core_id();
	int8_t log2Core = LOG2(NUM_CORES);

	uint32_t ch_chunk = (C >> log2Core) + ((C & (NUM_CORES - 1)) != 0);
	uint32_t ch_start = MIN(ch_chunk * core_id, C);
	uint32_t ch_end = MIN(ch_start + ch_chunk, C);
	uint32_t ch_count = ch_end - ch_start;

	if (ch_count == 0) {
		return;
	}

		const uint32_t HC = H * W * C;
		for (uint32_t n = 0; n < N; ++n) {
			const uint32_t n_base = n * HC;
			for (uint32_t c = ch_start; c < ch_end; ++c) {
				const float32_t beta_val = beta[c];
				const float32_t thr_val = threshold[c];
				for (uint32_t h = 0; h < H; ++h) {
					for (uint32_t w = 0; w < W; ++w) {
						const uint32_t idx = n_base + ((h * W + w) * C + c);
						const float32_t v = beta_val * mem_in[idx] + input[idx];
						const float32_t spk = v >= thr_val ? 1.0f : 0.0f;
						spike_out[idx] = spk;
						// reset-to-zero on spike
						mem_out[idx] = spk > 0.0f ? 0.0f : v;
					}
				}
			}
		}
}

