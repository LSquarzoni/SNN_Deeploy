# ----------------------------------------------------------------------
#
# File: InsertRequantShift.py
#
# Last edited: 16.12.2024
#
# Copyright (C) 2024, ETH Zurich and University of Bologna.
#
# Author: GitHub Copilot
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

import numpy as np
import onnx_graphsurgeon as gs

from Deeploy.CommonExtensions.OptimizationPasses.PassClasses import ReplaceSequentialPatternPass, contextagnostic


@contextagnostic
class InsertRequantShiftAfterConvPass(ReplaceSequentialPatternPass):
    """
    Inserts RequantShift nodes after Conv operations to convert int32 to int8.
    
    In quantized networks, Conv operations output int32 (accumulated products),
    but subsequent operations like LIF expect int8 inputs. This pass automatically
    inserts RequantShift nodes to perform this bit-width conversion.
    
    Pattern:
        Before: Conv(int8, int8 → int32) → LIF(expects int8)
        After:  Conv(int8, int8 → int32) → RequantShift(int32 → int8) → LIF(int8)
    
    The RequantShift node performs: output = clip((input * mul) >> add, -128, 127)
    where mul and add are quantization parameters computed from the layer's scale.
    """

    def __init__(self):
        # Create pattern: Conv node
        graph = gs.Graph()
        _input = gs.Variable(name='input_0')
        weights = gs.Variable(name='weights')
        conv_out = graph.layer(inputs=[_input, weights], outputs=['conv_out'], op='Conv', name='conv')
        graph.outputs.append(conv_out)
        graph.inputs.append(_input)

        name = "_INSERT_REQUANT_AFTER_CONV_PASS"
        super().__init__(graph, _insert_requant_after_conv_fun, name)


@contextagnostic
class InsertRequantShiftAfterAddPass(ReplaceSequentialPatternPass):
    """
    Inserts RequantShift nodes after Add operations to convert int32 to int8.
    
    In quantized networks, Add operations can output int32, but subsequent 
    operations like LIF expect int8 inputs. This pass automatically inserts 
    RequantShift nodes to perform this bit-width conversion.
    
    Pattern:
        Before: Add(int8, int8 → int32) → LIF(expects int8)
        After:  Add(int8, int8 → int32) → RequantShift(int32 → int8) → LIF(int8)
    """

    def __init__(self):
        # Create pattern: Add node
        graph = gs.Graph()
        input1 = gs.Variable(name='input_0')
        input2 = gs.Variable(name='input_1')
        add_out = graph.layer(inputs=[input1, input2], outputs=['add_out'], op='Add', name='add')
        graph.outputs.append(add_out)
        graph.inputs.extend([input1, input2])

        name = "_INSERT_REQUANT_AFTER_ADD_PASS"
        super().__init__(graph, _insert_requant_after_conv_fun, name)  # Reuse same function


def _insert_requant_after_conv_fun(graph: gs.Graph, match, name: str):
    """
    Insert RequantShift node after Conv operation.
    
    This function checks if the Conv output is int32 (indicating quantized convolution),
    and if so, inserts a RequantShift node to convert to int8.
    """
    matched_nodes = [m for k, m in match.nodes_map.items()]
    conv_node = matched_nodes[0]
    
    # Check if Conv already has a RequantShift after it
    conv_output = conv_node.outputs[0] if conv_node.outputs else None
    if not conv_output:
        return graph
    
    # Check consumers - if any is already a RequantShift, skip
    consumers = [n for n in graph.nodes if conv_output in n.inputs]
    if any(c.op == 'RequantShift' for c in consumers):
        return graph
    
    # CRITICAL: Only insert RequantShift for quantized networks
    # Skip fp32 networks by checking if the graph has any Quant/Dequant nodes
    has_quant_nodes = any(n.op in ['Quant', 'Dequant'] for n in graph.nodes)
    
    # Also check if Conv has quantized weights (int8 dtype)
    has_int8_weights = False
    if len(conv_node.inputs) >= 2:  # Conv has at least data and weight inputs
        weight = conv_node.inputs[1]
        if hasattr(weight, 'values') and weight.values is not None:
            has_int8_weights = weight.values.dtype in [np.int8, np.uint8]
    
    # Skip if this appears to be a fp32 network
    if not has_quant_nodes and not has_int8_weights:
        return graph
    
    # Check if output is a graph output - don't insert RequantShift before final output
    if conv_output in graph.outputs:
        return graph
    
    # Create new intermediate variable for RequantShift output
    # Copy shape from conv output to preserve tensor dimensions
    rqs_output = gs.Variable(
        name=f"{conv_output.name}_requantized", 
        dtype=np.int8,
        shape=conv_output.shape
    )
    
    # Create RequantShift node attributes
    # Note: Attributes must be gs.Constant or np.ndarray for the parser
    rqs_attrs = {
        'n_levels_out': gs.Constant(f"{name}_n_levels", values=np.array([256], dtype=np.int64)),
        'signed': gs.Constant(f"{name}_signed", values=np.array([1], dtype=np.int64)),
        'div': gs.Constant(f"{name}_div", values=np.array([1], dtype=np.int64)),
    }
    
    # Create constant inputs for RequantShift (mul, add)
    # These are placeholders - actual values will be computed from quantization scales
    mul_const = gs.Constant(name=f"{name}_mul", values=np.array([1], dtype=np.int32))
    add_const = gs.Constant(name=f"{name}_add", values=np.array([0], dtype=np.int32))
    
    rqs_node = gs.Node(
        op='RequantShift',
        name=f"{name}_{conv_node.name}_rqs",
        inputs=[conv_output, mul_const, add_const],
        outputs=[rqs_output],
        attrs=rqs_attrs
    )
    
    # Redirect all consumers to use RequantShift output
    for consumer in consumers:
        for i, inp in enumerate(consumer.inputs):
            if inp == conv_output:
                consumer.inputs[i] = rqs_output
    
    # Update conv output dtype to int32
    conv_output.dtype = np.int32
    
    # Add RequantShift node to graph
    graph.nodes.append(rqs_node)
    graph.cleanup().toposort()
    
    return graph
