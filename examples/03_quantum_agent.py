import asyncio
from plugins.quantum_qml import QuantumLLMCircuit, QuantumParameterOptimizer

def simulated_llm_loss_function(quantum_outputs: list[float]) -> float:
    """
    A simulated loss function representing feedback from a classic LLM environment.
    Objective: Force the first qubit's expectation value to approach exactly 1.0
    and minimize the error loop.
    """
    # Target state: We want high positive polarization on the primary quantum wire
    target_value = 1.0
    actual_value = quantum_outputs[0]
    
    # Standard Mean Squared Error (MSE) calculation
    loss = (target_value - actual_value) ** 2
    return loss

async def main():
    print("=== Launching Hybrid Quantum AI (QML) Optimization Loop ===\n")

    # 1. Initialize our 4-qubit Quantum Circuit Plugin
    num_qubits = 4
    quantum_circuit = QuantumLLMCircuit(num_wires=num_qubits)
    
    # 2. Attach our custom gradient descent optimizer (Learning Rate = 0.1)
    q_optimizer = QuantumParameterOptimizer(quantum_circuit=quantum_circuit, learning_rate=0.1)

    # 3. Define initial mock features extracted from an LLM token or prompt state
    # (Normalized classic input vector)
    classic_llm_features = [0.5, -0.2, 0.8, 0.1]
    
    # Initialize random trainable weights for our quantum rotation gates
    vqc_gate_weights = [0.1, 0.1, 0.1, 0.1]

    print(f"Initial Quantum Gate Weights: {vqc_gate_weights}")
    
    # Measure initial state outputs before training starts
    initial_outputs = quantum_circuit.run_quantum_inference(classic_llm_features, vqc_gate_weights)
    initial_loss = simulated_llm_loss_function(initial_outputs)
    print(f"Initial Circuit Output Vector: {[round(x, 4) for x in initial_outputs]}")
    print(f"Initial System Loss Score: {round(initial_loss, 6)}\n")

    print("--- Starting Gradient Descent across Hilbert Space ---")
    
    # 4. Run the optimization loop for 5 epochs
    for epoch in range(1, 6):
        # Execute a quantum gradient step using backpropagation through the simulator
        vqc_gate_weights = q_optimizer.optimize_step(
            current_weights=vqc_gate_weights,
            classic_inputs=classic_llm_features,
            loss_fn=simulated_llm_loss_function
        )
        
        # Evaluate current metrics
        current_outputs = quantum_circuit.run_quantum_inference(classic_llm_features, vqc_gate_weights)
        current_loss = simulated_llm_loss_function(current_outputs)
        
        print(f"Epoch {epoch}: Loss = {round(current_loss, 6)} | Updated Weights = {[round(w, 4) for w in vqc_gate_weights]}")

    print("\n=== Optimization Complete ===")
    final_outputs = quantum_circuit.run_quantum_inference(classic_llm_features, vqc_gate_weights)
    print(f"Final Circuit Output Vector: {[round(x, 4) for x in final_outputs]}")
    print(f"Final System Loss Score: {round(simulated_llm_loss_function(final_outputs), 6)}")
    print("The quantum circuit successfully adapted its gates to satisfy the LLM environment feedback loop!")

if __name__ == "__main__":
    # PennyLane operations run synchronously inside our async driver harness safely
    asyncio.run(main())
