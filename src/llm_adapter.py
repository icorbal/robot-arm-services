"""LLM provider abstraction with OpenAI implementation."""

import base64
import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class LLMAdapter(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        images: list[bytes] | None = None,
    ) -> dict[str, Any]:
        """Generate a JSON response from the LLM.

        Args:
            system_prompt: System/instruction prompt
            user_prompt: User message
            images: Optional list of PNG image bytes to include as vision input.

        Returns:
            Parsed JSON response from the LLM
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources."""
        ...


class OpenAIAdapter(LLMAdapter):
    """OpenAI LLM adapter."""

    def __init__(self, model: str = "gpt-4o", api_key_env: str = "OPENAI_API_KEY"):
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(
                f"API key not found. Set the {api_key_env} environment variable."
            )

        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        logger.info(f"OpenAI adapter initialized with model: {model}")

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        images: list[bytes] | None = None,
    ) -> dict[str, Any]:
        """Generate a JSON response using OpenAI API.

        When images are provided, they are encoded as base64 data URLs and
        included in the user message content blocks (vision API format).
        """
        logger.debug(f"Generating response with {self._model}")
        logger.debug(f"System prompt length: {len(system_prompt)}")
        logger.debug(f"User prompt length: {len(user_prompt)}")

        # Build user content: text-only or multimodal (text + images)
        if images:
            logger.debug(f"Including {len(images)} image(s) in request")
            user_content: list[dict[str, Any]] = [
                {"type": "text", "text": user_prompt},
            ]
            for img_bytes in images:
                b64 = base64.b64encode(img_bytes).decode("ascii")
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                })
            user_message: dict[str, Any] = {"role": "user", "content": user_content}
        else:
            user_message = {"role": "user", "content": user_prompt}

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    user_message,
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=2048,
            )

            content = response.choices[0].message.content
            if content is None:
                raise ValueError("Empty response from LLM")

            logger.debug(f"LLM response: {content[:200]}...")
            result = json.loads(content)
            return result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            raise ValueError(f"LLM returned invalid JSON: {e}") from e
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise

    async def close(self) -> None:
        """Close the OpenAI client."""
        await self._client.close()


def create_llm_adapter(
    provider: str = "openai",
    model: str = "gpt-4o",
    api_key_env: str = "OPENAI_API_KEY",
) -> LLMAdapter:
    """Factory function to create an LLM adapter.

    Args:
        provider: LLM provider name (currently only "openai")
        model: Model name
        api_key_env: Environment variable for API key

    Returns:
        LLM adapter instance
    """
    if provider == "openai":
        return OpenAIAdapter(model=model, api_key_env=api_key_env)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
