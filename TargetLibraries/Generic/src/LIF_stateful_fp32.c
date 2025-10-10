/* =====================================================================
 * Title:        LIF_stateful_fp32.c
 * Description:  Stateful Leaky Integrate-and-Fire neuron update (fp32, NCHW)
 *
 * Date:         10.10.2025
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

#include "DeeployBasicMath.h"
#include "kernel/LIF.h"

// Stateful LIF neuron update (fp32, NCHW).
// Semantics:
// - Uses mem_in if provided for this call; otherwise uses mem_state as the previous membrane.
// - Computes updated_mem = beta[c] * prev_mem + input.
// - Spike if updated_mem >= threshold[c].
// - On spike: spike_out = 1, mem_state = 0; else: spike_out = 0, mem_state = updated_mem.
// - mem_state is thus the one-cycle-late membrane, persisted across invocations.
void LIF_stateful_fp32(const float32_t *__restrict__ input,
                       const float32_t *__restrict__ mem_in,  // may be NULL
                       const float32_t *__restrict__ beta,
                       const float32_t *__restrict__ threshold,
                       float32_t *__restrict__ spike_out,
                       float32_t *__restrict__ mem_state,
                       uint32_t N, uint32_t C, uint32_t H, uint32_t W) {

    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t c = 0; c < C; ++c) {
            const float32_t beta_val = beta[c];
            const float32_t thr_val  = threshold[c];
            for (uint32_t h = 0; h < H; ++h) {
                for (uint32_t w = 0; w < W; ++w) {
                    const uint32_t idx = ((n * C + c) * H + h) * W + w;

                    const float32_t prev_mem = mem_in ? mem_in[idx] : mem_state[idx];
                    const float32_t updated_mem = beta_val * prev_mem + input[idx];

                    if (updated_mem >= thr_val) {
                        spike_out[idx] = 1.0f;
                        mem_state[idx] = 0.0f; // reset after spike
                    } else {
                        spike_out[idx] = 0.0f;
                        mem_state[idx] = updated_mem;
                    }
                }
            }
        }
    }
}
