# ----------------------------------------------------------------------
#
# File: LIFstatefulTileConstraint.py
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

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import uint16_t
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.TilingExtension.TilingCodegen import (
    AbsoluteHyperRectangle,
    HyperRectangle,
    TilingSchedule,
    VariableReplacementScheme,
)


class LIFstatefulTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        # Buffers: inputs (data_in, optional mem_in, beta, threshold), outputs (spike_out)
        in_data = parseDict['data_in']
        out_spike = parseDict['spike_out']

        # Add dims for tensors we tile (beta/threshold are 1D params and don't affect geometry)
        tilerModel.addTensorDimToModel(ctxt, in_data)
        tilerModel.addTensorDimToModel(ctxt, out_spike)

        # mem_in may be absent; constrain it only if present
        if 'mem_in' in parseDict and parseDict['mem_in'] is not None:
            in_mem = parseDict['mem_in']
            tilerModel.addTensorDimToModel(ctxt, in_mem)

            inputShape = ctxt.lookup(in_data).shape
            # Elementwise equality across input and output (NHWC order in PULP stack)
            for dim in range(len(inputShape)):
                in_d = tilerModel.getTensorDimVar(tensorName=in_data, dimIdx=dim)
                in_m = tilerModel.getTensorDimVar(tensorName=in_mem, dimIdx=dim)
                o_s = tilerModel.getTensorDimVar(tensorName=out_spike, dimIdx=dim)

                tilerModel.addConstraint(in_d == in_m)
                tilerModel.addConstraint(o_s == in_d)
        else:
            # Just tie output to input when no mem_in is provided
            inputShape = ctxt.lookup(in_data).shape
            for dim in range(len(inputShape)):
                in_d = tilerModel.getTensorDimVar(tensorName=in_data, dimIdx=dim)
                o_s = tilerModel.getTensorDimVar(tensorName=out_spike, dimIdx=dim)
                tilerModel.addConstraint(o_s == in_d)

        return tilerModel

    @staticmethod
    def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        # Keep channel dimension (last, NHWC) intact to match per-channel params (beta/threshold)
        in_data = parseDict['data_in']
        out_spike = parseDict['spike_out']

        shape = ctxt.lookup(in_data).shape
        numDims = len(shape)

        # Fix C dimension to full size; allow tiler to split N/H/W as needed
        in_c_var = tilerModel.getTensorDimVar(tensorName=in_data, dimIdx=numDims - 1)
        out_c_var = tilerModel.getTensorDimVar(tensorName=out_spike, dimIdx=numDims - 1)
        tilerModel.addConstraint(in_c_var == shape[-1])
        tilerModel.addConstraint(out_c_var == shape[-1])

        return tilerModel

    @classmethod
    def serializeTilingSolution(
        cls,
        tilingSolution: NodeMemoryConstraint,
        absoluteOutputCubes: List[AbsoluteHyperRectangle],
        targetMemLevel: str,
        ctxt: NetworkContext,
        operatorRepresentation: OperatorRepresentation,
    ) -> Tuple[VariableReplacementScheme, TilingSchedule]:

        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]

        # Addresses: inputs (data_in, optional mem_in), outputs (spike_out)
        addrNames = ['data_in', 'spike_out']
        # Include mem_in if present in operatorRepresentation
        if 'mem_in' in operatorRepresentation and operatorRepresentation['mem_in'] is not None:
            addrNames.insert(1, 'mem_in')

        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(
            tilingSolution, targetMemLevel, operatorRepresentation, addrNames
        )

        # Replacement variables N, C, H, W (even though PULP kernel expects NHWC, we expose these for templates)
        replacements: Dict[str, List[int]] = {k: [] for k in ['N', 'C', 'H', 'W']}
        replacementTypes = {k: PointerClass(uint16_t) for k in ['N', 'C', 'H', 'W']}

        # Build input/output load schedules
        inputLoadSchedule: List[Dict[str, HyperRectangle]] = []
        outputLoadSchedule: List[Dict[str, HyperRectangle]] = []

        for out_cube in outputCubes:
            # NHWC ordering for cubes in PULP stack (dims: [N, H, W, C])
            n_t, h_t, w_t, c_t = out_cube.dims

            # replacements used by the code template
            replacements['N'].append(n_t)
            replacements['C'].append(c_t)
            replacements['H'].append(h_t)
            replacements['W'].append(w_t)

            # Inputs have same offsets/dims as outputs for elementwise op
            data_in_cube = HyperRectangle(offset=out_cube.offset, dims=out_cube.dims)

            if 'mem_in' in operatorRepresentation and operatorRepresentation['mem_in'] is not None:
                mem_in_cube = HyperRectangle(offset=out_cube.offset, dims=out_cube.dims)
                inputLoadSchedule.append({'data_in': data_in_cube, 'mem_in': mem_in_cube})
            else:
                inputLoadSchedule.append({'data_in': data_in_cube})

            outputLoadSchedule.append({'spike_out': out_cube})

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)

        return variableReplacementSchedule, tilingSchedule
