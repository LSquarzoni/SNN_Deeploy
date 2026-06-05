# ----------------------------------------------------------------------
#
# File: iLIFTemplate.py
#
# Last edited: 15.12.2025
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

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class _iLIFTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        data_in = ctxt.lookup(operatorRepresentation['data_in'])
        mem_in = ctxt.lookup(operatorRepresentation['mem_in'])
        spike_out = ctxt.lookup(operatorRepresentation['spike_out'])
        mem_out = ctxt.lookup(operatorRepresentation['mem_out'])

        # Calculate offsets for quantization
        operatorRepresentation['input_offset'] = 0
        if hasattr(data_in, "_signed") and hasattr(data_in, "nLevels"):
            operatorRepresentation['input_offset'] = (data_in._signed == 0) * int(data_in.nLevels / 2)

        operatorRepresentation['mem_in_offset'] = 0
        if hasattr(mem_in, "_signed") and hasattr(mem_in, "nLevels"):
            operatorRepresentation['mem_in_offset'] = (mem_in._signed == 0) * int(mem_in.nLevels / 2)

        operatorRepresentation['output_offset'] = 0
        if hasattr(spike_out, "_signed") and hasattr(spike_out, "nLevels"):
            operatorRepresentation['output_offset'] = -(spike_out._signed == 0) * int(spike_out.nLevels / 2)

        operatorRepresentation['mem_out_offset'] = 0
        if hasattr(mem_out, "_signed") and hasattr(mem_out, "nLevels"):
            operatorRepresentation['mem_out_offset'] = -(mem_out._signed == 0) * int(mem_out.nLevels / 2)

        # Calculate scale factors for dequantization
        # Scale = (max - min) / nLevels
        operatorRepresentation['input_scale'] = 1.0
        if hasattr(data_in, "nLevels"):
            operatorRepresentation['input_scale'] = 1.0 / data_in.nLevels if data_in.nLevels > 0 else 1.0

        operatorRepresentation['mem_scale'] = 1.0
        if hasattr(mem_in, "nLevels"):
            operatorRepresentation['mem_scale'] = 1.0 / mem_in.nLevels if mem_in.nLevels > 0 else 1.0

        return ctxt, operatorRepresentation, []


referenceTemplate = _iLIFTemplate("""
// iLIF (Name: ${nodeName}, Op: ${nodeOp})
// Int8 activations with fp32 beta/threshold
<%
chw = C * H * W
batchOffset = C * H * W
%>
BEGIN_SINGLE_CORE
    ${data_in_type.typeName}   ref_${spike_out}_${data_in}      = ${data_in};
    ${mem_in_type.typeName}    ref_${spike_out}_${mem_in}       = ${mem_in};
    ${spike_out_type.typeName} ref_${spike_out}_${spike_out}    = ${spike_out};
    ${mem_out_type.typeName}   ref_${spike_out}_${mem_out}      = ${mem_out};
    for (uint32_t n=0; n<${N}; ++n) {
        LIF_s8_s8_s32(
            ref_${spike_out}_${data_in},
            ref_${spike_out}_${mem_in},
            ${beta},
            ${threshold},
            ref_${spike_out}_${spike_out},
            ref_${spike_out}_${mem_out},
            ${input_offset},
            ${mem_in_offset},
            ${output_offset},
            ${mem_out_offset},
            ${input_scale}f,
            ${mem_scale}f,
            1, ${C}, ${H}, ${W}
        );
        ref_${spike_out}_${data_in}   += ${batchOffset};
        ref_${spike_out}_${spike_out} += ${batchOffset};
        ref_${spike_out}_${mem_in}    += ${batchOffset};
        ref_${spike_out}_${mem_out}   += ${batchOffset};
    }
END_SINGLE_CORE
""")
