# ----------------------------------------------------------------------
#
# File: LIFTemplate.py
#
# Last edited: 02.10.2025
#
# Copyright (C) 2025, ETH Zurich and University of Bologna.
#
# Authors:
# - Lorenzo Squarzoni, University of Bologna
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// LIF (PULP) (Name: ${nodeName}, Op: ${nodeOp})
// Expects tensors: data_in, mem_in, beta, threshold
// Produces tensors: spike_out, mem_out
PULP_LIF_fp32_fp32(
    ${data_in},
    ${mem_in},
    ${beta},
    ${threshold},
    ${spike_out},
    ${mem_out},
    ${N}, ${C}, ${H}, ${W}
);
""")
