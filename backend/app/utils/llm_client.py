"""
LLM client wrapper
Unified API calls using OpenAI format.
Supports OpenAI-compatible APIs and Claude Code CLI.
"""

import json
import os
import re
from typing import Optional, Dict, Any, List
from openai import OpenAI

from ..config import Config


def create_llm_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    timeout: float = 300.0
):
    """
    Factory: returns ClaudeCodeClient when LLM_PROVIDER=claude-code,
    otherwise returns the standard LLMClient.
    """
    if Config.LLM_PROVIDER == 'claude-code':
        from .claude_code_client import ClaudeCodeClient
        return ClaudeCodeClient(model=model, timeout=timeout)
    return LLMClient(api_key=api_key, base_url=base_url, model=model, timeout=timeout)


def create_smart_llm_client(timeout: float = 300.0):
    """
    Factory for intelligence-sensitive workflows (reports, ontology, graph reasoning).
    Uses SMART_* config when set, otherwise falls back to the default LLM client.
    """
    if not Config.SMART_MODEL_NAME:
        return create_llm_client(timeout=timeout)

    provider = Config.SMART_PROVIDER or Config.LLM_PROVIDER

    if provider == 'claude-code':
        from .claude_code_client import ClaudeCodeClient
        return ClaudeCodeClient(model=Config.SMART_MODEL_NAME, timeout=timeout)

    return LLMClient(
        api_key=Config.SMART_API_KEY or Config.LLM_API_KEY,
        base_url=Config.SMART_BASE_URL or Config.LLM_BASE_URL,
        model=Config.SMART_MODEL_NAME,
        timeout=timeout,
    )


class LLMClient:
    """LLM client using OpenAI-compatible APIs"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 300.0
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model = model or Config.LLM_MODEL_NAME

        if not self.api_key:
            raise ValueError("LLM_API_KEY is not configured")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=timeout,
            default_headers={
                'HTTP-Referer': 'https://github.com/aaronjmars/MiroShark',
                'X-Title': 'MiroShark',
                'X-OpenRouter-Categories': 'roleplay',
            },
        )

        # Ollama context window size — prevents prompt truncation.
        self._num_ctx = int(os.environ.get('OLLAMA_NUM_CTX', '8192'))

    def _is_ollama(self) -> bool:
        """Check if we're talking to an Ollama server."""
        return '11434' in (self.base_url or '')

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None
    ) -> str:
        """
        Send a chat request

        Args:
            messages: List of messages
            temperature: Temperature parameter
            max_tokens: Maximum number of tokens
            response_format: Response format (e.g., JSON mode)

        Returns:
            Model response text
        """
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if response_format:
            kwargs["response_format"] = response_format

        # For Ollama: pass num_ctx via extra_body to prevent prompt truncation
        if self._is_ollama() and self._num_ctx:
            kwargs["extra_body"] = {
                "options": {"num_ctx": self._num_ctx}
            }

        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        # Some models (e.g., MiniMax M2.5) include <think> reasoning content in the content field, which needs to be removed
        content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
        return content

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """
        Send a chat request and return JSON

        Args:
            messages: List of messages
            temperature: Temperature parameter
            max_tokens: Maximum number of tokens

        Returns:
            Parsed JSON object
        """
        response = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )
        # Clean up markdown code block markers
        cleaned_response = response.strip()
        cleaned_response = re.sub(r'^```(?:json)?\s*\n?', '', cleaned_response, flags=re.IGNORECASE)
        cleaned_response = re.sub(r'\n?```\s*$', '', cleaned_response)
        cleaned_response = cleaned_response.strip()

        # Surface enough detail when vLLM returns empty / truncated / malformed
        # output so upstream callers (e.g. NER extractor) don't just silently
        # record 0 entities on every chunk. An empty response in particular
        # usually means a hermes-parser failure or context-length overflow
        # rather than an actual JSON parse issue.
        if not cleaned_response:
            raise ValueError(
                f"LLM returned empty response (raw_len={len(response)}, "
                f"raw_repr={response[:200]!r})"
            )
        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON from LLM: {e} | cleaned_len={len(cleaned_response)} "
                f"preview={cleaned_response[:300]!r}"
            )
