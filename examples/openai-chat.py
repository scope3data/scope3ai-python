from openai import OpenAI, AsyncOpenAI
import asyncio
from typing import AsyncGenerator, Generator
import os
from dotenv import load_dotenv

from scope3ai._scope3ai import Scope3AI

# Load environment variables from .env file if it exists
load_dotenv()

# Get API key from environment variables
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")
scope3_api_key = os.getenv('SCOPE3_API_KEY')
if not scope3_api_key:
    raise ValueError("SCOPE3_API_KEY environment variable is not set")

# Initialize the clients
client = OpenAI()  # Synchronous client
async_client = AsyncOpenAI()  # Asynchronous client

# 1. Synchronous, Non-streaming Chatbot
def sync_non_streaming_chat(prompt: str) -> str:
    """Simple synchronous chat without streaming"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        stream=False
    )
    return response.choices[0].message.content

# 2. Synchronous, Streaming Chatbot
def sync_streaming_chat(prompt: str) -> Generator[str, None, None]:
    """Synchronous chat with streaming responses"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content

# 3. Asynchronous, Non-streaming Chatbot
async def async_non_streaming_chat(prompt: str) -> str:
    """Asynchronous chat without streaming"""
    response = await async_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        stream=False
    )
    return response.choices[0].message.content

# 4. Asynchronous, Streaming Chatbot
async def async_streaming_chat(prompt: str) -> AsyncGenerator[str, None]:
    """Asynchronous chat with streaming responses"""
    response = await async_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    
    async for chunk in response:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content

# Example usage for each implementation
def main():
    scope3 = Scope3AI(api_key=scope3_api_key)

    prompt = "Tell me a short joke"
    
    try:
        # 1. Sync Non-streaming
        print("\n1. Synchronous Non-streaming:")
        response = sync_non_streaming_chat(prompt)
        print(response)
        
        # 2. Sync Streaming
        print("\n2. Synchronous Streaming:")
        for chunk in sync_streaming_chat(prompt):
            print(chunk, end='', flush=True)
        print()  # New line after streaming completes
        
        # 3 & 4. Async examples need to be run in an async context
        async def run_async_examples():
            try:
                # 3. Async Non-streaming
                print("\n3. Asynchronous Non-streaming:")
                response = await async_non_streaming_chat(prompt)
                print(response)
                
                # 4. Async Streaming
                print("\n4. Asynchronous Streaming:")
                async for chunk in async_streaming_chat(prompt):
                    print(chunk, end='', flush=True)
                print()  # New line after streaming completes
            except Exception as e:
                print(f"An error occurred in async operation: {str(e)}")
        
        # Run async examples
        asyncio.run(run_async_examples())
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
