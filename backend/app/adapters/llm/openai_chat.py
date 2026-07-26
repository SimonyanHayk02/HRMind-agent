from __future__ import annotations

from openai import AsyncOpenAI


class OpenAIChatLLM:
    def __init__(self, *, api_key: str, model: str) -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    async def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.0,
        response_json: bool = False,
    ) -> str:
        text, _ = await self.complete_with_usage(
            system=system, user=user, temperature=temperature, response_json=response_json
        )
        return text

    async def complete_with_usage(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.0,
        response_json: bool = False,
    ) -> tuple[str, dict[str, int]]:
        kwargs = {}
        if response_json:
            kwargs["response_format"] = {"type": "json_object"}
        resp = await self._client.chat.completions.create(
            model=self._model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **kwargs,
        )
        text = resp.choices[0].message.content or ""
        usage = {
            "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(resp.usage, "completion_tokens", 0) or 0,
            "total_tokens": getattr(resp.usage, "total_tokens", 0) or 0,
        }
        return text, usage
