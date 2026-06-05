# ----------------------------------------------------------------------
#
# File: LIFTileConstraint.py
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

from typing import Dict, List, Tuple

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import uint16_t
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel, PerformanceHint
from Deeploy.TilingExtension.TilingCodegen import (
	AbsoluteHyperRectangle,
	HyperRectangle,
	TilingSchedule,
	VariableReplacementScheme,
)


class LIFTileConstraint(TileConstraint):

	@staticmethod
	def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
		# Buffers: inputs (data_in, mem_in, beta, threshold), outputs (spike_out, mem_out)
		in_data = parseDict['data_in']
		in_mem = parseDict['mem_in']
		beta = parseDict['beta']
		threshold = parseDict['threshold']
		out_spike = parseDict['spike_out']
		out_mem = parseDict['mem_out']

		# Add dims for all tensors including per-channel params
		for name in [in_data, in_mem, beta, threshold, out_spike, out_mem]:
			tilerModel.addTensorDimToModel(ctxt, name)

		inputShape = ctxt.lookup(in_data).shape
		numDims = len(inputShape)

		# Elementwise equality across inputs/outputs (NHWC order in PULP stack)
		for dim in range(numDims):
			in_d = tilerModel.getTensorDimVar(tensorName=in_data, dimIdx=dim)
			in_m = tilerModel.getTensorDimVar(tensorName=in_mem, dimIdx=dim)
			o_s = tilerModel.getTensorDimVar(tensorName=out_spike, dimIdx=dim)
			o_m = tilerModel.getTensorDimVar(tensorName=out_mem, dimIdx=dim)

			tilerModel.addConstraint(in_d == in_m)
			tilerModel.addConstraint(o_s == in_d)
			tilerModel.addConstraint(o_m == in_d)

		# Link beta and threshold to the channel dimension (last dim in NHWC)
		# These can be 1D [C] or 3D [C,1,1] tensors
		# Only constrain them to channel dimension if they're NOT scalars
		beta_shape = ctxt.lookup(beta).shape
		threshold_shape = ctxt.lookup(threshold).shape
		
		# Detect scalars: either [] or [1] or [1,1,1]
		beta_is_scalar = len(beta_shape) == 0 or (len(beta_shape) >= 1 and beta_shape[0] == 1)
		threshold_is_scalar = len(threshold_shape) == 0 or (len(threshold_shape) >= 1 and threshold_shape[0] == 1)
		
		in_c_var = tilerModel.getTensorDimVar(tensorName=in_data, dimIdx=numDims - 1)
		beta_var = tilerModel.getTensorDimVar(tensorName=beta, dimIdx=0)
		threshold_var = tilerModel.getTensorDimVar(tensorName=threshold, dimIdx=0)
		
		# Only link non-scalar params to channel dimension for tiling
		# Scalars stay fixed at size 1 regardless of channel tiling
		if not beta_is_scalar:
			tilerModel.addConstraint(beta_var == in_c_var)
		else:
			tilerModel.addConstraint(beta_var == 1)
			
		if not threshold_is_scalar:
			tilerModel.addConstraint(threshold_var == in_c_var)
		else:
			tilerModel.addConstraint(threshold_var == 1)
		
		# For 3D tensors [C,1,1], constrain dimensions 1 and 2 to be fixed at 1
		if len(beta_shape) == 3:
			beta_dim1 = tilerModel.getTensorDimVar(tensorName=beta, dimIdx=1)
			beta_dim2 = tilerModel.getTensorDimVar(tensorName=beta, dimIdx=2)
			tilerModel.addConstraint(beta_dim1 == 1)
			tilerModel.addConstraint(beta_dim2 == 1)
			
		if len(threshold_shape) == 3:
			threshold_dim1 = tilerModel.getTensorDimVar(tensorName=threshold, dimIdx=1)
			threshold_dim2 = tilerModel.getTensorDimVar(tensorName=threshold, dimIdx=2)
			tilerModel.addConstraint(threshold_dim1 == 1)
			tilerModel.addConstraint(threshold_dim2 == 1)

		return tilerModel

	@staticmethod
	def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
		# LIF is an elementwise operation that can be tiled in all dimensions (N, H, W, C).
		# 
		# NOTE: Due to a framework limit, spatial tiling (H/W) will produce INCORRECT results
		# because DMA stride calculation is hardcoded for channel-only tiling. However, this
		# configuration is used for performance benchmarking to measure inference cycles.
		#
		# CONSTRAINTS:
		# - Minimum tile sizes prevent excessive tiling overhead and code generation issues
		# - Channel tiling must use multiples of 8 for efficient 8-core parallelization
		# - Spatial dimensions have minimum sizes to avoid degenerate tiles
		
		in_data = parseDict['data_in']
		shape = ctxt.lookup(in_data).shape
		numDims = len(shape)

		# Spatial dimension constraints: minimum 4x4 tiles to avoid tiny tiles
		# These minimums balance memory usage vs tiling overhead
		h_var = tilerModel.getTensorDimVar(tensorName=in_data, dimIdx=1)
		w_var = tilerModel.getTensorDimVar(tensorName=in_data, dimIdx=2)
		
		tilerModel.addConstraint(h_var >= 4)  # Minimum height per tile
		tilerModel.addConstraint(w_var >= 4)  # Minimum width per tile

		# Channel dimension constraints: minimum 8 channels for parallelization
		c_var = tilerModel.getTensorDimVar(tensorName=in_data, dimIdx=numDims - 1)
		total_channels = shape[-1]
		
		if total_channels < 8:
			# Very small channel count: keep all together
			tilerModel.addConstraint(c_var == total_channels)
		else:
			# Allow channel tiling with minimum 8 channels per tile
			tilerModel.addConstraint(c_var >= 8)
			# Require multiples of 8 for optimal core distribution (8 cores on PULP)
			tilerModel.addConstraint((c_var % 8) == 0)

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

		# Include all inputs (data_in, mem_in, beta, threshold) and outputs (spike_out, mem_out)
		addrNames = ['data_in', 'mem_in', 'beta', 'threshold', 'spike_out', 'mem_out']
		inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(
			tilingSolution, targetMemLevel, operatorRepresentation, addrNames
		)

		# For LIF template we need N, C, H, W variables.
		# PULP tensors are NHWC, and per-channel params (beta, threshold) are indexed by C.
		replacements: Dict[str, List[int]] = {k: [] for k in ['N', 'C', 'H', 'W']}
		replacementTypes = {k: PointerClass(uint16_t) for k in ['N', 'C', 'H', 'W']}

		# Build input/output load schedules
		inputLoadSchedule: List[Dict[str, HyperRectangle]] = []
		outputLoadSchedule: List[Dict[str, HyperRectangle]] = []

		# DEBUG: Print tiling information
		#print(f"\n[LIF TILING DEBUG] Node: {operatorRepresentation.get('nodeName', 'unknown')}")
		#print(f"  Number of tiles: {len(outputCubes)}")
		
		for tile_idx, out_cube in enumerate(outputCubes):
			# Extract both offset and dims for NHWC ordering
			(n_off, h_off, w_off, c_off) = out_cube.offset
			(n_t, h_t, w_t, c_t) = out_cube.dims

			# DEBUG: Print tile info
			#print(f"  Tile {tile_idx}: offset=({n_off},{h_off},{w_off},{c_off}) dims=({n_t},{h_t},{w_t},{c_t})")

			# Template replacements: use the TILED dimensions for this cube
			replacements['N'].append(n_t)
			replacements['C'].append(c_t)  # Use tiled C, not full C
			replacements['H'].append(h_t)
			replacements['W'].append(w_t)

			# Inputs have same offsets/dims as outputs for elementwise op
			data_in_cube = HyperRectangle(offset=out_cube.offset, dims=out_cube.dims)
			mem_in_cube = HyperRectangle(offset=out_cube.offset, dims=out_cube.dims)
			
			# beta and threshold: Handle tiling for per-channel parameters
			# These can be 1D [C] or 3D [C,1,1] tensors indexed by channel dimension
			beta_shape = ctxt.lookup(operatorRepresentation['beta']).shape
			threshold_shape = ctxt.lookup(operatorRepresentation['threshold']).shape
			
			# Get number of channels (first dimension)
			beta_channels = 1 if len(beta_shape) == 0 or beta_shape[0] == 1 else beta_shape[0]
			threshold_channels = 1 if len(threshold_shape) == 0 or threshold_shape[0] == 1 else threshold_shape[0]
			
			# Create HyperRectangles matching the tensor dimensionality
			# For [C,1,1] tensors: tile only the channel dimension, keep trailing dims full
			# For [C] tensors: create 1D rectangle
			# When accessing full tensor (c_t >= channels), must use offset 0
			if len(beta_shape) == 3:
				# 3D tensor [C,1,1]: tile channel, keep trailing dims
				if beta_channels == 1 or c_t >= beta_channels:
					beta_cube = HyperRectangle(offset=(0, 0, 0), dims=(beta_channels, 1, 1))
				else:
					beta_cube = HyperRectangle(offset=(c_off, 0, 0), dims=(c_t, 1, 1))
			else:
				# 1D tensor [C]: tile channel dimension only
				if beta_channels == 1 or c_t >= beta_channels:
					beta_cube = HyperRectangle(offset=(0,), dims=(beta_channels,))
				else:
					beta_cube = HyperRectangle(offset=(c_off,), dims=(c_t,))
			
			if len(threshold_shape) == 3:
				# 3D tensor [C,1,1]: tile channel, keep trailing dims
				if threshold_channels == 1 or c_t >= threshold_channels:
					threshold_cube = HyperRectangle(offset=(0, 0, 0), dims=(threshold_channels, 1, 1))
				else:
					threshold_cube = HyperRectangle(offset=(c_off, 0, 0), dims=(c_t, 1, 1))
			else:
				# 1D tensor [C]: tile channel dimension only
				if threshold_channels == 1 or c_t >= threshold_channels:
					threshold_cube = HyperRectangle(offset=(0,), dims=(threshold_channels,))
				else:
					threshold_cube = HyperRectangle(offset=(c_off,), dims=(c_t,))

			inputLoadSchedule.append({
				'data_in': data_in_cube, 
				'mem_in': mem_in_cube,
				'beta': beta_cube,
				'threshold': threshold_cube
			})
			outputLoadSchedule.append({'spike_out': out_cube, 'mem_out': out_cube})

		#print(f"[LIF TILING DEBUG] Replacements: N={replacements['N']}, C={replacements['C']}, H={replacements['H']}, W={replacements['W']}\n")

		tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
		variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)

		return variableReplacementSchedule, tilingSchedule

