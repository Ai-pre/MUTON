from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from openai import OpenAI


DEFAULT_BASE_URL = "http://127.0.0.1:8000/b/v1"
DEFAULT_MODEL_NAME = "track_b_model"
DEFAULT_SYSTEM_PROMPT = (
    "You are a multimodal dialogue understanding assistant. "
    "Use the face image, speech audio, and transcript together, and answer in one concise Korean sentence."
)
DEFAULT_USER_INSTRUCTION = (
    "얼굴 이미지, 음성, 대사를 함께 참고해서 화자의 감정, 태도, 상황을 한국어 한 문장으로 설명해라."
)


def create_client(base_url: str, api_key: str = "not-needed") -> OpenAI:
    return OpenAI(base_url=base_url.rstrip("/"), api_key=api_key)


def audio_url_to_payload(audio_url: str) -> dict[str, Any]:
    suffix = Path(audio_url).suffix.lower().lstrip(".") or "wav"
    if suffix == "wave":
        suffix = "wav"
    encoded = base64.b64encode(audio_url.encode("utf-8")).decode("utf-8")
    return {
        "type": "input_audio",
        "input_audio": {
            "data": encoded,
            "format": suffix,
        },
    }


def build_messages(
    image_url: str,
    audio_url: str,
    script: str,
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    instruction: str = DEFAULT_USER_INSTRUCTION,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    if image_url:
        content.append({"type": "image_url", "image_url": {"url": image_url}})
    if audio_url:
        content.append(audio_url_to_payload(audio_url))
    content.append(
        {
            "type": "text",
            "text": f"{instruction}\n\n대사: {script.strip()}",
        }
    )
    return [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": content,
        },
    ]


def request_summary(
    *,
    client: OpenAI,
    model_name: str,
    image_url: str,
    audio_url: str,
    script: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    instruction: str = DEFAULT_USER_INSTRUCTION,
    max_tokens: int = 96,
) -> str:
    messages = build_messages(
        image_url=image_url,
        audio_url=audio_url,
        script=script,
        system_prompt=system_prompt,
        instruction=instruction,
    )
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"skip_reasoning": True}},
    )
    text = response.choices[0].message.content or ""
    return text.strip()
