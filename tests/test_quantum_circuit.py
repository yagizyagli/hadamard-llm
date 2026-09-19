import os
import sys
import pytest

# Enforce project root injection before pulling framework modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Dynamic import guard for isolated CI/CD environments missing heavy simulation suites
try:
    import pennylane as qml
    from plugins.quantum_qml.circuit import QuantumLLMCircuit
    from plugins.quantum_qml.optimizer import QuantumParameterOptimizer
    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False


@pytest.mark.skipif(not QUANTUM_AVAILABLE, reason="Quantum simulator dependencies (PennyLane) not installed in this environment.")
def test_quantum_circuit_execution_bounds():
    """Tests if the VQC runs and outputs normalized expectation values between -1.0 and 1.0."""
    circuit = QuantumLLMCircuit(num_wires=3)
    mock_inputs = [0.1, 0.5, -0.9]
    mock_weights = [0.2, 0.2, 0.2]
    
    outputs = circuit.run_quantum_inference(mock_inputs, mock_weights)
    
    assert len(outputs) == 3
    for val in outputs:
        assert -1.0 <= val <= 1.0


@pytest.mark.skipif(not QUANTUM_AVAILABLE, reason="Quantum simulator dependencies (PennyLane) not installed in this environment.")
def test_quantum_optimizer_step():
    """Tests if a single training optimization step correctly adjusts the variational gate weights."""
    circuit = QuantumLLMCircuit(num_wires=2)
    optimizer = QuantumParameterOptimizer(quantum_circuit=circuit, learning_rate=0.1)
    
    initial_weights = [0.5, 0.5]
    inputs = [0.1, 0.2]
    
    def mock_loss(quantum_outputs):
        return sum(quantum_outputs)
        
    updated_weights = optimizer.optimize_step(initial_weights, inputs, mock_loss)
    
    assert len(updated_weights) == 2
    assert updated_weights != initial_weights
