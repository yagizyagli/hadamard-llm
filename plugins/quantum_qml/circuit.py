from typing import Any, List
import pennylane as qml
from pennylane import numpy as np


class QuantumLLMCircuit:
    """
    A foundational Variational Quantum Circuit (VQC) plugin designed to preprocess,
    encode, or optimize classic LLM parameters/prompt weights via quantum states.
    """

    def __init__(self, num_wires: int = 4):
        self.num_wires = num_wires
        # Define a standard simulator device (can be switched to a real hardware backend later)
        self.device = qml.device("default.qubit", wires=self.num_wires)
        self._init_qnode()

    def _init_qnode(self):
        """
        Internal helper to construct and bind the runnable Quantum Node (QNode).
        """
        @qml.qnode(self.device)
        def _circuit(inputs: List[float], weights: List[float]):
            # Step 1: Put all qubits into a superposition state using Hadamard gates
            for i in range(self.num_wires):
                qml.Hadamard(wires=i)

            # Step 2: Encode classic LLM inputs into quantum states via Angle Embedding (RY rotation)
            for i in range(min(len(inputs), self.num_wires)):
                qml.RY(inputs[i], wires=i)

            # Step 3: Entangle qubits to process relationships (similar to self-attention mechanism)
            for i in range(self.num_wires - 1):
                qml.CNOT(wires=[i, i + 1])
            if self.num_wires > 2:
                qml.CNOT(wires=[self.num_wires - 1, 0])

            # Step 4: Strongly Entangling Layers with trainable weights
            for i in range(self.num_wires):
                qml.RZ(weights[i % len(weights)], wires=i)

            # Step 5: Measure the expected value (PauliZ) of each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_wires)]

        self.qnode = _circuit

    def run_quantum_inference(self, classic_inputs: List[float], trainable_weights: List[float]) -> List[float]:
        """
        Executes the quantum circuit simulator with classic vector inputs and weights.
        Returns a processed list of quantum expectation values (-1.0 to 1.0).
        """
        # Convert raw Python lists to PennyLane-optimized NumPy arrays
        inputs_np = np.array(classic_inputs, requires_grad=False)
        weights_np = np.array(trainable_weights, requires_grad=True)
        
        # Execute the quantum graph
        quantum_results = self.qnode(inputs_np, weights_np)
        return [float(val) for val in quantum_results]
