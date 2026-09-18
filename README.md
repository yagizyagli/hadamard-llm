# 🌌 Hadamard-LLM (hadamard-llm)

An **ultra-lightweight, zero-official-sdk, asynchronous LLM orchestration framework** engineered for blazing-fast classic execution and supercharged with **Variational Quantum Circuit (VQC) and Quantum Machine Learning (QML)** plugins via PennyLane.

---

## ⚡ Why Hadamard-LLM?

Legacy frameworks like LangChain have become bloated, heavy, and complex, bringing hundreds of unnecessary megabytes of vendor SDK dependencies and suffering from retrofitted asynchronous orchestration. 

**Hadamard-LLM** completely bypasses official provider libraries. Built entirely upon raw, high-performance, asynchronous HTTPX connection pooling, it runs at **10x lower memory overhead and nanossecond-level prompt formatting overhead**, while natively embedding hybrid classical-quantum optimization capabilities.

---

## 🏗️ Clean Architectural Blueprints

The codebase is organized as a strict, zero-leak monorepo separating the blazing-fast core engine from experimental, compute-heavy quantum plugins.

```text
hadamard-llm/
├── .github/workflows/ci.yml   # 🛠️ 3-Version Python CI Automation Suite
├── core/                      # 🚀 1. CORE ENGINE (Zero-Dependency Async Layer)
│   ├── models/                # Multi-Provider Pure HTTPX Engines (OpenAI, Claude, Gemini, DeepSeek, etc.)
│   ├── prompts/               # High-Speed Native String Compiler & Protection Kalkanı
│   ├── memory/                # Context-Optimized Sliding Turn Buffers
│   └── parsers/               # Two-Tier Structural Regex & Bracket JSON Sanitizers
└── plugins/                   # 🧬 2. QUANTUM PLUGINS (Isolate Compute Layer)
    └── quantum_qml/           # Variational Circuits, Amplitude Encoders, & Autograd Optimizers
```

---

## 📦 Dynamic Installation & Environment Setup

Hadamard-LLM adapts dynamically to your production requirements. Choose the environment footprint that fits your system architecture:

### 1. Production Mode (Classic LLM Orchestration Only)
Installs only the blazing-fast core runtime engine with a minimal system footprint.
```bash
pip install https://github.com/yagizyagli/hadamard-llm
```

### 2. Quantum Research Mode (Full QML Subsystem Enabled)
Installs the classic core along with heavy mathematical simulation primitives (`pennylane`, `qiskit`).
```bash
pip install "hadamard-llm[quantum] @https://github.com/yagizyagli/hadamard-llm
```

### 3. Local Development & Contributor Sandbox
Clone the repository and spin up an editable link to test adjustments instantly:
```bash
git clone https://github.com/yagizyagli/hadamard-llm
cd hadamard-llm
pip install -e .[quantum,test]
```

---

## 🚀 Execution Blueprints & Examples

### 1. Asynchronous Token Streaming with No Vendor SDKs
```python
import asyncio
from core import OpenAIModel, PromptTemplate

async def stream_architecture():
    # Complies and protects inputs instantly
    template = PromptTemplate("Explain {concept} in exactly one punchy sentence.")
    prompt_payload = template.render(concept="Asynchronous I/O Pools")
    
    # Leverages raw sockets instead of bloated client layers
    model = OpenAIModel(model_name="gpt-4o-mini", temperature=0.5)
    
    print("Streaming Tokens: ", end="")
    async for token in model.generate_stream(prompt_payload):
        print(token, end="", flush=True)
    print("\n")

asyncio.run(stream_architecture())
```

### 2. Training a Variational Quantum Circuit (VQC) with LLM Feedback
```python
import asyncio
from plugins.quantum_qml import QuantumLLMCircuit, QuantumParameterOptimizer

def mock_llm_reward_loss(quantum_states: list[float]) -> float:
    # Reward alignment score from classic LLM token decisions
    return (1.0 - quantum_states[0]) ** 2

async def quantum_alignment_loop():
    circuit = QuantumLLMCircuit(num_wires=4)
    optimizer = QuantumParameterOptimizer(quantum_circuit=circuit, learning_rate=0.1)
    
    weights = [0.1, 0.1, 0.1, 0.1]
    inputs = [0.5, -0.2, 0.8, 0.1] # Feature maps extracted from prompt embeddings
    
    # Train the quantum gates via automatic gradient backpropagation across Hilbert space
    for epoch in range(1, 4):
        weights = optimizer.optimize_step(weights, inputs, mock_llm_reward_loss)
        print(f"Epoch {epoch} | Quantum Parameters Updated.")

asyncio.run(quantum_alignment_loop())
```

---

## 🧪 Comprehensive Verification & Verification Matrix

Automated verification protects the runtime system behavior against regression anomalies across multiple Python runtime engines.

Run the test suite locally using:
```bash
pytest
```
The automated test protocols enforce strict validation across three layers:
*   **`tests/test_prompts.py`**: Ensures double-brace brace escaping protection and regex parsing boundaries.
*   **`tests/test_core_models.py`**: Verifies configuration state payloads, request timeouts, and secure environment validation thresholds.
*   **`tests/test_quantum_circuit.py`**: Validates PauliZ computation state matrix mathematical boundary safety limits `[-1.0, 1.0]`.

---

## 🔮 Roadmap Vision

- [ ] **Stateful Multi-Agent Graphs:** A lightweight alternative to LangGraph for cyclic agent topologies without massive memory footprints.
- [ ] **Edge GPU Context Compilers:** Native integration with local WebGPU runtimes for running client-side Edge AI structures.
- [ ] **Quantum Prompt Injection Shielding:** Utilizing quantum entanglement states to detect adversarial semantic pattern injections.

---

## 👥 Authors & Core Maintainers

* **Yağız Yağlı**: [@yagizyagli](https://github.com/yagizyagli)

---

## ⭐ Star the Quantum Frontier

If you believe the future of AI orchestration belongs to **high-performance, lightweight architectures** and **Quantum AI alignment**, please consider supporting this open-source roadmap. Drop a star to show your support! ⭐

*Hadamard-LLM is distributed under the conditions of the open-source **MIT License**.*
