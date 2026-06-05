/* =====================================================================
 * Title:        LIF_s8.c
 * Description:  Leaky Integrate-and-Fire neuron update (int8, NCHW)
 *
 * Date:         15.12.2025
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

// LIF neuron update with int8 quantization.
// Input/output tensors are int8, but beta and threshold remain fp32 (as exported from ONNX).
// Intermediate computations use int32/float32 to prevent overflow.
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
// - input      : [N, C, H, W] input current (int8)
// - mem_in     : [N, C, H, W] previous membrane potential (int8)
// - beta       : [C] decay factor per channel (fp32, from ONNX)
// - threshold  : [C] firing threshold per channel (fp32, from ONNX)
// - spike_out  : [N, C, H, W] output spikes (int8)
// - mem_out    : [N, C, H, W] updated membrane potential (int8)

void LIF_s8_s8_s32(const int8_t *__restrict__ input,
                   const int8_t *__restrict__ mem_in,
                   const float32_t *__restrict__ beta,
                   const float32_t *__restrict__ threshold,
                   int8_t *__restrict__ spike_out,
                   int8_t *__restrict__ mem_out, int32_t input_offset,
                   int32_t mem_in_offset, int32_t output_offset,
                   int32_t mem_out_offset, float32_t input_scale,
                   float32_t mem_scale, uint32_t N, uint32_t C, uint32_t H,
                   uint32_t W) {

  for (uint32_t n = 0; n < N; ++n) {
    for (uint32_t c = 0; c < C; ++c) {
      float32_t beta_val = beta[c];
      float32_t thr_val = threshold[c];
      for (uint32_t h = 0; h < H; ++h) {
        for (uint32_t w = 0; w < W; ++w) {
          uint32_t idx = ((n * C + c) * H + h) * W + w;

          // Dequantize membrane and apply decay
          float32_t mem_real =
              ((int32_t)mem_in[idx] + mem_in_offset) * mem_scale;
          float32_t decayed_mem = beta_val * mem_real;

          // Dequantize input and add to decayed membrane
          float32_t input_real =
              ((int32_t)input[idx] + input_offset) * input_scale;
          float32_t updated_mem = decayed_mem + input_real;

          // Check threshold and generate spike
          if (updated_mem >= thr_val) {
            // Spike: output 1 (quantized), reset membrane to 0 (quantized)
            spike_out[idx] = (int8_t)(1 - output_offset);
            mem_out[idx] = (int8_t)(0 - mem_out_offset);
          } else {
            // No spike: output 0 (quantized), store updated membrane
            // (quantized)
            spike_out[idx] = (int8_t)(0 - output_offset);
            // Quantize updated_mem back to int8
            int32_t mem_quantized =
                (int32_t)(updated_mem / mem_scale) - mem_out_offset;
            // Clamp to int8 range
            if (mem_quantized > 127)
              mem_quantized = 127;
            if (mem_quantized < -128)
              mem_quantized = -128;
            mem_out[idx] = (int8_t)mem_quantized;
          }
        }
      }
    }
  }
}
