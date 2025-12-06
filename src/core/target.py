"""
OpenRouter Target Interface for A-Red-Team-Havoc
Handles communication with target LLMs via OpenRouter API
Uses OpenAI client with OpenRouter base URL
"""

import os
import asyncio
from typing import Dict, Optional, Any, List
from dataclasses import dataclass
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI


@dataclass
class TargetResponse:
    """Represents a response from the target model"""
    success: bool
    response_text: str
    model: str
    prompt: str
    template_name: str
    timestamp: str
    latency_ms: float
    error: Optional[str] = None
    raw_response: Optional[Dict] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "response_text": self.response_text,
            "model": self.model,
            "prompt": self.prompt,
            "template_name": self.template_name,
            "timestamp": self.timestamp,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "raw_response": self.raw_response
        }


class OpenRouterTarget:
    """
    Target interface for OpenRouter API
    Uses OpenAI client with OpenRouter base URL
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "meta-llama/llama-3.1-8b-instruct",
        max_retries: int = 3
    ):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not provided or found in environment")

        self.model = model
        self.max_retries = max_retries

        # Initialize OpenAI client with OpenRouter base URL
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
            default_headers={
                "HTTP-Referer": "https://github.com/nia-red-team",
                "X-Title": "HAVOC Red Team Toolkit"
            }
        )

        # Thread pool for async execution
        self._executor = ThreadPoolExecutor(max_workers=20)

    def _send_request(
        self,
        prompt: str,
        template_name: str = "direct",
        system_prompt: Optional[str] = None
    ) -> TargetResponse:
        """
        Send a single prompt to the target model (synchronous)
        """
        start_time = datetime.now()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages
                )

                latency = (datetime.now() - start_time).total_seconds() * 1000
                response_text = response.choices[0].message.content

                return TargetResponse(
                    success=True,
                    response_text=response_text,
                    model=self.model,
                    prompt=prompt,
                    template_name=template_name,
                    timestamp=datetime.now().isoformat(),
                    latency_ms=latency,
                    raw_response={
                        "id": response.id,
                        "model": response.model,
                        "usage": dict(response.usage) if response.usage else None
                    }
                )

            except Exception as e:
                latency = (datetime.now() - start_time).total_seconds() * 1000
                error_msg = str(e)

                if attempt == self.max_retries - 1:
                    return TargetResponse(
                        success=False,
                        response_text="",
                        model=self.model,
                        prompt=prompt,
                        template_name=template_name,
                        timestamp=datetime.now().isoformat(),
                        latency_ms=latency,
                        error=error_msg
                    )

                # Exponential backoff
                import time
                time.sleep(2 ** attempt)

    def send_prompt(
        self,
        prompt: str,
        template_name: str = "direct",
        system_prompt: Optional[str] = None
    ) -> TargetResponse:
        """Send a single prompt synchronously"""
        return self._send_request(prompt, template_name, system_prompt)

    async def send_prompt_async(
        self,
        prompt: str,
        template_name: str = "direct",
        system_prompt: Optional[str] = None
    ) -> TargetResponse:
        """Send a single prompt asynchronously using thread pool"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._send_request,
            prompt,
            template_name,
            system_prompt
        )

    async def send_batch_async(
        self,
        prompts: List[Dict[str, str]],
        concurrency: int = 10
    ) -> List[TargetResponse]:
        """
        Send multiple prompts concurrently

        Args:
            prompts: List of dicts with 'prompt' and 'template_name' keys
            concurrency: Max concurrent requests

        Returns:
            List of TargetResponse objects
        """
        semaphore = asyncio.Semaphore(concurrency)

        async def limited_request(item: Dict[str, str]) -> TargetResponse:
            async with semaphore:
                return await self.send_prompt_async(
                    prompt=item['prompt'],
                    template_name=item.get('template_name', 'direct'),
                    system_prompt=item.get('system_prompt')
                )

        tasks = [limited_request(item) for item in prompts]
        return await asyncio.gather(*tasks)

    def send_batch(
        self,
        prompts: List[Dict[str, str]],
        concurrency: int = 10
    ) -> List[TargetResponse]:
        """Synchronous wrapper for send_batch_async"""
        return asyncio.run(self.send_batch_async(prompts, concurrency))

    def set_model(self, model: str) -> None:
        """Change the target model"""
        self.model = model

    def get_available_models(self) -> List[str]:
        """
        Common OpenRouter models for red teaming
        (Not an exhaustive list - OpenRouter supports many more)
        """
        return [
            # Meta Llama
            "meta-llama/llama-3.1-8b-instruct",
            "meta-llama/llama-3.1-70b-instruct",
            "meta-llama/llama-3.1-405b-instruct",
            "meta-llama/llama-3.2-3b-instruct",
            "meta-llama/llama-3.3-70b-instruct",
            # Mistral
            "mistralai/mistral-7b-instruct",
            "mistralai/mixtral-8x7b-instruct",
            "mistralai/mistral-large",
            # Google
            "google/gemini-pro-1.5",
            "google/gemini-flash-1.5",
            # Anthropic
            "anthropic/claude-3-haiku",
            "anthropic/claude-3.5-sonnet",
            "anthropic/claude-3-opus",
            # OpenAI
            "openai/gpt-4o-mini",
            "openai/gpt-4o",
            "openai/gpt-4-turbo",
            # Qwen
            "qwen/qwen-2.5-72b-instruct",
            # DeepSeek
            "deepseek/deepseek-chat",
            "deepseek/deepseek-r1",
        ]


# Test utility
if __name__ == "__main__":
    import sys

    target = OpenRouterTarget()
    print(f"Target model: {target.model}")

    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
        print(f"\nSending: {prompt}\n")
        response = target.send_prompt(prompt)
        if response.success:
            print(f"Response: {response.response_text}")
            print(f"Latency: {response.latency_ms:.2f}ms")
        else:
            print(f"Error: {response.error}")
