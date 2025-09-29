# ----------------------------------------------------------------------
#
# File: FloatLIFTemplate.py
#
# Last edited: 25.09.2025
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
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// LIF (Name: ${nodeName}, Op: ${nodeOp})
// Expects tensors: input (current), mem_in (prev membrane), beta, threshold
// Produces tensors: spike_out, mem_out
<%
# total size of one batch for C,H,W
chw = C * H * W
batchOffset = C * H * W
%>
BEGIN_SINGLE_CORE
    ${data_in_type.typeName}   ref_in      = ${data_in};
    ${mem_in_type.typeName}    ref_mem_in  = ${mem_in};
    ${spike_out_type.typeName} ref_spike   = ${spike_out};
    ${mem_out_type.typeName}   ref_mem_out = ${mem_out};
    for (uint32_t n=0; n<${N}; ++n) {
        LIF_fp32_fp32(
            ref_in,
            ref_mem_in,
            ${beta},
            ${threshold},
            ref_spike,
            ref_mem_out,
            1, ${C}, ${H}, ${W}
        );
        ref_in += ${batchOffset};
        ref_spike += ${batchOffset};
        ref_mem_in += ${batchOffset};
        ref_mem_out += ${batchOffset};
    }
END_SINGLE_CORE
""")
