# ----------------------------------------------------------------------
#
# File: LIFstatefulTemplate.py
#
# Last edited: 10.10.2025
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

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class _PULPLIFStatefulTemplate(NodeTemplate):

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        # Hoist a persistent membrane state buffer per node, same shape as input
        data_in_buf = ctxt.lookup(operatorRepresentation['data_in'])
        mem_name = operatorRepresentation['spike_out'] + "_mem_state"

        # Create VariableBuffer for state; keep it local so standard alloc/dealloc applies
        mem_state_buf = ctxt.VariableBuffer(name=mem_name, shape=list(data_in_buf.shape))
        # Add to context first so type annotation can resolve references properly
        ctxt.add(mem_state_buf, 'local')
        # Propagate type info from input (float32*)
        if hasattr(data_in_buf, "_type") and data_in_buf._type is not None:
            ctxt.annotateType(mem_state_buf.name, data_in_buf._type)

        # Expose the name to the code template
        operatorRepresentation['mem_state'] = mem_state_buf.name

        return ctxt, operatorRepresentation, []


referenceTemplate = _PULPLIFStatefulTemplate("""
// LIF stateful (PULP) (Name: ${nodeName}, Op: ${nodeOp})
// Expects tensors: input (current), [optional mem_in], beta, threshold
// Produces tensors: spike_out
<%
# total size of one batch for C,H,W
batchOffset = C * H * W
%>
BEGIN_SINGLE_CORE
    // One-time zero-init of persistent membrane state
    static uint8_t __lifstate_init_${mem_state} = 0;
    if (!__lifstate_init_${mem_state}) {
        for (uint32_t i = 0; i < (uint32_t)(${N})*(uint32_t)(${C})*(uint32_t)(${H})*(uint32_t)(${W}); ++i) {
            ${mem_state}[i] = 0.0f;
        }
        __lifstate_init_${mem_state} = 1;
    }
END_SINGLE_CORE

    // Ensure all cores see the initialized mem_state before computing
    pi_cl_team_barrier();

    // Run the multi-core kernel on all cores
    PULP_LIF_stateful_fp32_fp32(
        ${data_in},
        % if has_mem_in:
        ${mem_in},
        % else:
        NULL,
        % endif
        ${beta},
        ${threshold},
        ${spike_out},
        ${mem_state},
        ${N}, ${H}, ${W}, ${C}
    );
""")
