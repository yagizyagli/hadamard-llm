from typing import Callable, List
import pennylane as qml
from pennylane import numpy as np
from plugins.quantum_qml.circuit import QuantumLLMCircuit


class QuantumParameterOptimizer:
    """
    A high-performance optimization layer that tunes variational quantum circuit
    weights based on feedback loops from classic LLM outputs or downstream rewards.
    """

    def __init__(self, quantum_circuit: QuantumLLMCircuit, learning_rate: float = 0.05):
        self.qc = quantum_circuit
        self.lr = learning_rate
        # Initialize PennyLane's built-in gradient descent optimizer
        self.optimizer = qml.GradientDescentOptimizer(stepsize=self.lr)

    def optimize_step(
        self, 
        current_weights: List[float], 
        classic_inputs: List[float], 
        loss_fn: Callable[[List[float]], float]
    ) -> List[float]:
        """
        Executes a single gradient descent step to update variational quantum weights.
        
        Args:
            current_weights: The mutable parameter array of the quantum gates.
            classic_inputs: Static feature array fed into the angle embedding.
            loss_fn: A classic evaluation wrapper that measures system loss.
        """
        # Convert inputs and weights into target tensor types
        weights_np = np.array(current_weights, requires_grad=True)
        inputs_np = np.array(classic_inputs, requires_grad=False)

        # Define an isolated cost function compatible with PennyLane's autograd engine
        def cost_function(w):
            quantum_predictions = self.qc.qnode(inputs_np, w)
            # Evaluate predictions using the injected loss routine
            return loss_fn([float(x) for x in quantum_predictions])

        # Compute gradients and perform parameter update step
        updated_weights, _ = self.optimizer.step_and_cost(cost_function, weights_np)
        
        return [float(w) for w in updated_weights]
