/* =====================================================================
 * Title:        LIF_s8.c
 * Description:  Leaky Integrate-and-Fire neuron update (int8, NHWC) for PULP
 *
 * Date:         16.12.2025
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

#include "DeeployPULPMath.h"

// LIF neuron update with int8 quantization for PULP (NHWC layout).
// Input/output tensors are int8, but beta and threshold remain fp32.
// 
// Quantization parameters:
// - input_offset: zero-point for input current
// - mem_in_offset: zero-point for membrane potential  
// - output_offset: zero-point for spike output
// - mem_out_offset: zero-point for membrane output
// - input_scale: scale factor for input
// - mem_scale: scale factor for membrane
//
// Formula (per element):
//   mem_real = (mem_in[idx] + mem_in_offset) * mem_scale
//   input_real = (input[idx] + input_offset) * input_scale
//   updated_mem_real = beta[c] * mem_real + input_real
//   if (updated_mem_real >= threshold[c]):
//     spike_out[idx] = 1 - output_offset (quantized spike)
//     mem_out[idx] = 0 - mem_out_offset (quantized reset)
//   else:
//     spike_out[idx] = 0 - output_offset (quantized no spike)
//     mem_out[idx] = quantize(updated_mem_real)
//
// Dimensions:
// - input      : [N, H, W, C] input current (int8, NHWC)
// - mem_in     : [N, H, W, C] previous membrane potential (int8, NHWC)
// - beta       : [C] decay factor per channel (fp32)
// - threshold  : [C] firing threshold per channel (fp32)
// - spike_out  : [N, H, W, C] output spikes (int8, NHWC)
// - mem_out    : [N, H, W, C] updated membrane potential (int8, NHWC)

void LIF_s8_s8_NHWC_s32(const int8_t *__restrict__ input,
                        const int8_t *__restrict__ mem_in,
                        const float32_t *__restrict__ beta,
                        const float32_t *__restrict__ threshold,
                        int8_t *__restrict__ spike_out,
                        int8_t *__restrict__ mem_out, int32_t input_offset,
                        int32_t mem_in_offset, int32_t output_offset,
                        int32_t mem_out_offset, float32_t input_scale,
                        float32_t mem_scale, uint32_t N, uint32_t C, uint32_t H,
                        uint32_t W) {

  uint32_t core_id = pi_core_id();
  uint32_t n_cores = NUM_CORES;
  uint32_t total_elements = N * H * W * C;
  uint32_t chunk_size = (total_elements + n_cores - 1) / n_cores;
  uint32_t start = core_id * chunk_size;
  uint32_t end = (start + chunk_size > total_elements) ? total_elements : start + chunk_size;

  for (uint32_t idx = start; idx < end; ++idx) {
    // NHWC layout: idx = ((n * H + h) * W + w) * C + c
    uint32_t c = idx % C;
    
    float32_t beta_val = beta[c];
    float32_t thr_val = threshold[c];

    // Dequantize membrane and apply decay
    float32_t mem_real = ((int32_t)mem_in[idx] + mem_in_offset) * mem_scale;
    float32_t decayed_mem = beta_val * mem_real;

    // Dequantize input and add to decayed membrane
    float32_t input_real = ((int32_t)input[idx] + input_offset) * input_scale;
    float32_t updated_mem = decayed_mem + input_real;

    // Check threshold and generate spike
    if (updated_mem >= thr_val) {
      // Spike: output 1 (quantized), reset membrane to 0 (quantized)
      spike_out[idx] = (int8_t)(1 - output_offset);
      mem_out[idx] = (int8_t)(0 - mem_out_offset);
    } else {
      // No spike: output 0 (quantized), store updated membrane (quantized)
      spike_out[idx] = (int8_t)(0 - output_offset);
      // Quantize updated_mem back to int8
      int32_t quant_mem = (int32_t)(updated_mem / mem_scale) - mem_out_offset;
      mem_out[idx] = (int8_t)CLAMP(quant_mem, -128, 127);
    }
  }
}
