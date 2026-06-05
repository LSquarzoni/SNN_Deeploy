/* =====================================================================
 * Title:        Tanh_s8.c
 * Description:  Hyperbolic tangent activation function (int8)
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

// Tanh activation with int8 quantization using lookup table interpolation.
// 
// The tanh function is approximated using a 256-entry lookup table (LUT)
// covering the full int8 input range [-128, 127]. For better accuracy,
// linear interpolation can be used between table entries.
//
// Quantization:
// - input values are int8, offset by input_offset
// - output values are int8, offset by output_offset
// - intermediate computation in int32
//
// input        : [size] input tensor (int8)
// output       : [size] output tensor (int8)
// size         : number of elements
// input_offset : zero-point for input
// output_offset: zero-point for output
// input_scale  : scale factor for dequantizing input (fixed-point: multiplier >> shift)
// output_scale : scale factor for quantizing output (fixed-point: multiplier >> shift)

void Tanh_s8_s8(int8_t *input, int8_t *output, int32_t size,
                int32_t input_offset, int32_t output_offset,
                int32_t input_scale_mult, int32_t input_scale_shift,
                int32_t output_scale_mult, int32_t output_scale_shift) {

  // Lookup table for tanh values: tanh_lut[i] corresponds to tanh(x)
  // where x is mapped from [-128, 127] to approximately [-4, 4]
  // tanh(x) ranges from -1 to 1, quantized to int8 range [-127, 127]
  static const int8_t tanh_lut[256] = {
      -127, -127, -127, -127, -127, -127, -127, -127, // -128 to -121
      -127, -127, -127, -127, -127, -127, -127, -127, // -120 to -113
      -127, -127, -127, -127, -127, -127, -127, -127, // -112 to -105
      -127, -127, -127, -127, -127, -127, -127, -127, // -104 to -97
      -127, -127, -127, -127, -127, -127, -127, -127, // -96 to -89
      -127, -127, -127, -127, -127, -127, -127, -126, // -88 to -81
      -126, -126, -126, -126, -125, -125, -125, -124, // -80 to -73
      -124, -123, -123, -122, -122, -121, -120, -119, // -72 to -65
      -118, -117, -116, -115, -114, -113, -111, -110, // -64 to -57
      -109, -107, -106, -104, -103, -101, -99, -97,   // -56 to -49
      -96, -94, -92, -90, -88, -85, -83, -81,         // -48 to -41
      -79, -76, -74, -71, -69, -66, -64, -61,         // -40 to -33
      -58, -55, -53, -50, -47, -44, -41, -38,         // -32 to -25
      -35, -32, -29, -26, -23, -20, -17, -14,         // -24 to -17
      -11, -8, -5, -2, 0, 3, 6, 9,                     // -16 to -9
      12, 15, 18, 21, 24, 27, 30, 33,                 // -8 to -1
      36, 39, 42, 45, 48, 51, 54, 57,                 // 0 to 7
      60, 63, 65, 68, 70, 73, 75, 77,                 // 8 to 15
      80, 82, 84, 86, 88, 90, 91, 93,                 // 16 to 23
      95, 96, 98, 99, 101, 102, 104, 105,             // 24 to 31
      106, 108, 109, 110, 111, 112, 113, 114,         // 32 to 39
      115, 116, 117, 118, 119, 120, 120, 121,         // 40 to 47
      122, 122, 123, 123, 124, 124, 125, 125,         // 48 to 55
      125, 126, 126, 126, 126, 126, 127, 127,         // 56 to 63
      127, 127, 127, 127, 127, 127, 127, 127,         // 64 to 71
      127, 127, 127, 127, 127, 127, 127, 127,         // 72 to 79
      127, 127, 127, 127, 127, 127, 127, 127,         // 80 to 87
      127, 127, 127, 127, 127, 127, 127, 127,         // 88 to 95
      127, 127, 127, 127, 127, 127, 127, 127,         // 96 to 103
      127, 127, 127, 127, 127, 127, 127, 127,         // 104 to 111
      127, 127, 127, 127, 127, 127, 127, 127,         // 112 to 119
      127, 127, 127, 127, 127, 127, 127, 127          // 120 to 127
  };

  for (int i = 0; i < size; i++) {
    // Dequantize input
    int32_t x_dequant = (int32_t)input[i] + input_offset;

    // Apply input scaling to map to tanh domain
    int32_t x_scaled = (x_dequant * input_scale_mult) >> input_scale_shift;

    // Clamp to int8 range for LUT indexing
    if (x_scaled > 127)
      x_scaled = 127;
    if (x_scaled < -128)
      x_scaled = -128;

    // Look up tanh value (offset by 128 for array indexing)
    int32_t tanh_value = tanh_lut[x_scaled + 128];

    // Apply output scaling and quantize
    int32_t y_scaled = (tanh_value * output_scale_mult) >> output_scale_shift;
    int32_t y_quantized = y_scaled - output_offset;

    // Clamp to int8 range
    if (y_quantized > 127)
      y_quantized = 127;
    if (y_quantized < -128)
      y_quantized = -128;

    output[i] = (int8_t)y_quantized;
  }
}
