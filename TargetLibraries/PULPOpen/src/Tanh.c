/* =====================================================================
 * Title:        Tanh.c
 * Description:  Hyperbolic tangent activation function for PULP platform
 *
 * Date:         17.10.2025
 *
 * ===================================================================== */

/*
 * Copyright (C) 2022 ETH Zurich and University of Bologna.
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
#include "pmsis.h"
#include <math.h>

void PULP_Tanh_fp32_fp32(float32_t *input, float32_t *output, uint32_t size) {

  int8_t core_id = pi_core_id();
  int8_t log2Core = LOG2(NUM_CORES);

  int32_t chunk = (size >> log2Core) + ((size & (NUM_CORES - 1)) != 0);
  int32_t start = MIN(chunk * core_id, size);
  int32_t end = MIN(start + chunk, size);
  int32_t local_size = end - start;

  float32_t *local_input = input + start;
  float32_t *local_output = output + start;

  for (int32_t i = 0; i < local_size; i++) {
    local_output[i] = tanhf(local_input[i]);
  }
}
