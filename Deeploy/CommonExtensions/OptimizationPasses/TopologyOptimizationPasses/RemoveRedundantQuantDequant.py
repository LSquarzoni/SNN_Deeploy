import onnx_graphsurgeon as gs

from Deeploy.CommonExtensions.OptimizationPasses.Matchers import Match
from Deeploy.CommonExtensions.OptimizationPasses.PassClasses import ReplaceSequentialPatternPass, contextagnostic


def _remove_quant_dequant_fun(graph: gs.Graph, match: Match, name: str):
    """
    Remove redundant QUANT→DEQUANT pairs that break integer dataflow.
    
    For proper quantized inference, we need:
    - INPUT → Quant (fp32 → int8)
    - [int8 operations]
    - Dequant → OUTPUT (int8 → fp32)
    
    This pass removes Quant→Dequant pairs but handles special cases:
    
    Case 1: INPUT → Quant → Dequant → ...
            Remove only the Dequant, keep Quant after input
            Result: INPUT → Quant → ...
    
    Case 2: ... → Quant → Dequant → OUTPUT
            Remove only the Quant, keep Dequant before output
            Result: ... → Dequant → OUTPUT
    
    Case 3: ... → Quant → Dequant → OP → ...
            Remove both (middle of network)
            Result: ... → OP → ...
    """
    matched_nodes = [m for k, m in match.nodes_map.items()]
    quant_node = matched_nodes[0]
    dequant_node = matched_nodes[1]

    # Get inputs and outputs
    quant_input = quant_node.inputs[0] if quant_node.inputs else None
    quant_output = quant_node.outputs[0] if quant_node.outputs else None
    dequant_output = dequant_node.outputs[0] if dequant_node.outputs else None
    
    if not quant_input or not quant_output or not dequant_output:
        return graph
    
    # Case 1: Quant is fed by graph input → Remove only Dequant
    if quant_input in graph.inputs:
        # Connect consumers to Quant output (skip Dequant)
        for consumer in [n for n in graph.nodes if dequant_output in n.inputs]:
            for i, inp in enumerate(consumer.inputs):
                if inp == dequant_output:
                    consumer.inputs[i] = quant_output
        
        # Remove only Dequant
        dequant_node.inputs.clear()
        dequant_node.outputs.clear()
        graph.nodes.remove(dequant_node)
        graph.cleanup().toposort()
        return graph
    
    # Case 2: Dequant feeds graph output → Remove only Quant
    if dequant_output in graph.outputs:
        # Connect Dequant to Quant's input (skip Quant)
        dequant_node.inputs[0] = quant_input
        
        # Remove only Quant
        quant_node.inputs.clear()
        quant_node.outputs.clear()
        graph.nodes.remove(quant_node)
        graph.cleanup().toposort()
        return graph
    
    # Case 3: Middle of network → Remove both Quant and Dequant
    for consumer in [n for n in graph.nodes if dequant_output in n.inputs]:
        for i, inp in enumerate(consumer.inputs):
            if inp == dequant_output:
                consumer.inputs[i] = quant_input
    
    # Remove both nodes
    quant_node.inputs.clear()
    quant_node.outputs.clear()
    graph.nodes.remove(quant_node)
    
    dequant_node.inputs.clear()
    dequant_node.outputs.clear()
    graph.nodes.remove(dequant_node)
    
    graph.cleanup().toposort()
    return graph


@contextagnostic
class RemoveRedundantQuantDequantPass(ReplaceSequentialPatternPass):
    """
    Removes redundant QUANT→DEQUANT sequences between operations.
    
    In fake-quantized models from QAT (Quantization-Aware Training), operations
    are surrounded by QUANT→DEQUANT pairs for training purposes. For deployment,
    these should be removed to enable true integer dataflow:
    
    Before: Conv(int8→int32) → Quant → Dequant → LIF(fp32→fp32)
    After:  Conv(int8→int32) → Requant(int32→int8) → LIF(int8→int8)
    
    This pass removes intermediate Quant→Dequant pairs while preserving:
    - Initial Quant at network input (fp32 → int8 conversion)
    - Final Dequant at network output (int8 → fp32 conversion)
    
    The result is an integer-only computation graph with Requant operations
    handling bit-width changes between layers.
    """

    def __init__(self):
        # Create pattern graph: Quant → Dequant
        graph = gs.Graph()
        _input = gs.Variable(name='input_0')
        quant_out = graph.layer(inputs=[_input], outputs=['quant_out'], op='Quant', name='quant')
        dequant_out = graph.layer(inputs=quant_out, outputs=['dequant_out'], op='Dequant', name='dequant')
        graph.outputs.append(dequant_out)
        graph.inputs.append(_input)

        name = "_REMOVE_REDUNDANT_QUANT_DEQUANT_PASS"
        super().__init__(graph, _remove_quant_dequant_fun, name)
