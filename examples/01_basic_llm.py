import asyncio
import os
from core import OpenAIModel, DeepSeekModel, PromptTemplate

async def main():
    # Ensure you have your API keys set in your environment
    # os.environ["OPENAI_API_KEY"] = "your-key-here"
    # os.environ["DEEPSEEK_API_KEY"] = "your-key-here"

    print("--- Initialize Prompt Template ---")
    # Define a clean, dynamic prompt template using our custom engine
    template = PromptTemplate(
        "Generate a highly technical and brief explanation of {concept} in computer science."
    )
    rendered_prompt = template.render(concept="Asynchronous I/O")
    print(f"Rendered Prompt: {rendered_prompt}\n")

    print("--- Scenario 1: Standard Generation (DeepSeek) ---")
    try:
        # Initializing our zero-dependency DeepSeek model wrapper
        model = DeepSeekModel(model_name="deepseek-chat", temperature=0.3)
        
        print("Sending request asynchronously...")
        response = await model.generate(
            prompt=rendered_prompt, 
            system_instruction="You are an elite systems architect. Be direct and concise."
        )
        print(f"Model Response:\n{response}\n")
    except Exception as e:
        print(f"Skipping Scenario 1 (Check your API keys): {e}\n")


    print("--- Scenario 2: Token Streaming (OpenAI) ---")
    try:
        # Initializing our zero-dependency OpenAI model wrapper
        openai_model = OpenAIModel(model_name="gpt-4o-mini", temperature=0.7)
        
        print("Streaming response tokens in real-time:")
        async for token in openai_model.generate_stream(
            prompt="Write a 3-sentence poetic metaphor about quantum superposition.",
            system_instruction="You are a poetic physicist."
        ):
            # Print each token as it arrives from the raw HTTPX socket connection
            print(token, end="", flush=True)
        print("\n\nStream Finished Successfully!")
    except Exception as e:
        print(f"\nSkipping Scenario 2 (Check your API keys): {e}")

if __name__ == "__main__":
    # Run the high-performance async event loop
    asyncio.run(main())
