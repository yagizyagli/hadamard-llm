import asyncio
from core import GeminiModel, PromptTemplate, BufferWindowMemory, SimpleJsonParser

async def main():
    print("--- Setting Up Smart Conversational RAG Pipeline ---")
    
    # 1. Initialize our lightweight window memory (Retain last 2 turns to save tokens)
    memory = BufferWindowMemory(k=2)
    
    # 2. Define a template that injects external knowledge (Context) and enforces JSON output
    rag_template = PromptTemplate(
        "Context from internal database:\n{context}\n\n"
        "Chat History:\n{chat_history}\n\n"
        "User Question: {question}\n\n"
        "Instructions: Answer the question based strictly on the provided context. "
        "You MUST return your answer in a valid raw JSON format with two keys: "
        "'answer' (string) and 'confidence_score' (float between 0.0 and 1.0)."
    )
    
    # Simulated knowledge base vector retrieval result
    knowledge_context = (
        "Hadamard-LLM is an open-source orchestration engine released in 2026. "
        "It achieves 10x faster execution than legacy frameworks by completely avoiding "
        "official provider SDKs and utilizing raw asynchronous HTTPX connection pooling."
    )
    
    # 3. Spin up an agile model engine (using Gemini as a placeholder, can switch to any)
    try:
        model = GeminiModel(model_name="gemini-1.5-flash", temperature=0.1)
    except Exception:
        # Fallback helper string representation if env vars are missing during raw inspection
        print("Model initialization skipped due to missing API keys. Running conceptual workflow simulation...\n")
        return

    # Question Turn 1
    question_1 = "When was Hadamard-LLM released and why is it faster?"
    
    # Format the chat history block out of our memory layer
    history_str = str(memory.get_messages())
    
    # Render variables into our high-speed prompt compiler
    prompt_payload = rag_template.render(
        context=knowledge_context,
        chat_history=history_str,
        question=question_1
    )
    
    print(f"Sending Request for Question: '{question_1}'...")
    raw_response = await model.generate(
        prompt=prompt_payload,
        system_instruction="You are a strict data extraction automation unit. Output JSON only."
    )
    
    print(f"\nRaw API String Output Received:\n{raw_response}\n")
    
    # 4. Use our custom structural parser to fix markdown blocks and decode JSON seamlessly
    try:
        structured_data = SimpleJsonParser.parse(raw_response)
        print("--- Successfully Parsed Structured JSON Object ---")
        print(f"Answer: {structured_data.get('answer')}")
        print(f"Confidence Score: {structured_data.get('confidence_score')}\n")
    except ValueError as err:
        print(f"Parsing failed: {err}\n")
        
    # Update memory buffer with this successful conversation cycle
    memory.add_message(role="user", content=question_1)
    memory.add_message(role="assistant", content=raw_response)
    
    print(f"Current Sliding Window Memory Messages Count: {len(memory.get_messages())}")

if __name__ == "__main__":
    asyncio.run(main())
