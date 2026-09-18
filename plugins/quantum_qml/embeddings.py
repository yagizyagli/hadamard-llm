from typing import List
import math
from plugins.quantum_qml.circuit import QuantumLLMCircuit


class QuantumEmbeddingsProcessor:
    """
    Transforms classic high-dimensional vector embeddings into quantum-enhanced
    state outputs using Variational Quantum Circuits.
    """

    def __init__(self, num_qubits: int = 4):
        self.num_qubits = num_qubits
        self.quantum_circuit = QuantumLLMCircuit(num_wires=num_qubits)
        # Default symmetric weights for the trainable layer of the quantum circuit
        self.default_weights = [math.pi / 4] * num_qubits

    def _normalize_to_angles(self, classic_vector: List[float]) -> List[float]:
        """
        Compresses and normalizes a high-dimensional classic vector into 
        rotation angles (-pi to pi) that fits the quantum circuit's wire constraints.
        """
        if not classic_vector:
            return [0.0] * self.num_qubits

        # Step 1: Chunk or pool the classic vector to match the number of qubits
        chunk_size = max(1, len(classic_vector) // self.num_qubits)
        pooled_vector = []
        
        for i in range(self.num_qubits):
            start = i * chunk_size
            end = start + chunk_size if i < self.num_qubits - 1 else len(classic_vector)
            chunk = classic_vector[start:end]
            # Average pooling of the chunk
            avg_val = sum(chunk) / len(chunk) if chunk else 0.0
            pooled_vector.append(avg_val)

        # Step 2: Normalize values to [-pi, pi] for quantum gate rotations (RY/RZ)
        max_val = max(max([abs(v) for v in pooled_vector]), 1e-5)
        angle_vector = [(v / max_val) * math.pi for v in pooled_vector]
        
        return angle_vector

    def process_embedding(self, classic_embedding: List[float], weights: Optional[List[float]] = None) -> List[float]:
        """
        Takes a classic embedding vector (e.g., from OpenAI/Ollama), processes it through
        the Quantum Circuit, and returns a quantum-enhanced, low-dimensional footprint.
        """
        # Encode classic vector into valid quantum rotation angles
        quantum_angles = self._normalize_to_angles(classic_embedding)
        
        # Determine weights to use
        circuit_weights = weights if weights and len(weights) == self.num_qubits else self.default_weights

        # Execute quantum interference transformation
        quantum_output = self.quantum_circuit.run_quantum_inference(
            classic_inputs=quantum_angles,
            trainable_weights=circuit_weights
        )
        
        return quantum_output
