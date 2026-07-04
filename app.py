import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class Settings:
    airops_execute_url = os.getenv("AIROPS_EXECUTE_URL", "").strip()
    airops_api_key = os.getenv("AIROPS_API_KEY", "").strip()
    airops_input_key = os.getenv("AIROPS_INPUT_KEY", "prompt").strip()
    proxy_api_key = os.getenv("PROXY_API_KEY", "").strip()
    exposed_model = os.getenv("EXPOSED_MODEL", "claude-opus-4.8").strip()
    request_timeout = float(os.getenv("AIROPS_TIMEOUT_SECONDS", "120"))
    include_raw_messages = env_bool("AIROPS_INCLUDE_RAW_MESSAGES", False)


settings = Settings()

app = FastAPI(title="AirOps SillyTavern Proxy", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str
    content: Any = ""


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage] = Field(default_factory=list)
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


def require_proxy_auth(authorization: Optional[str]) -> None:
    if not settings.proxy_api_key:
        return
    expected = f"Bearer {settings.proxy_api_key}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid proxy API key")


def require_airops_config() -> None:
    missing = []
    if not settings.airops_execute_url:
        missing.append("AIROPS_EXECUTE_URL")
    if not settings.airops_api_key:
        missing.append("AIROPS_API_KEY")
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Missing required environment variables: {', '.join(missing)}",
        )


def flatten_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif item.get("type") == "image_url":
                    image_url = item.get("image_url", {})
                    if isinstance(image_url, dict):
                        parts.append(f"[image: {image_url.get('url', '')}]")
                    else:
                        parts.append(f"[image: {image_url}]")
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p)
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def messages_to_prompt(messages: List[ChatMessage]) -> str:
    lines = []
    for message in messages:
        role = (message.role or "user").upper()
        text = flatten_content(message.content).strip()
        if text:
            lines.append(f"{role}: {text}")
    lines.append("ASSISTANT:")
    return "\n\n".join(lines)


def build_airops_payload(req: ChatCompletionRequest) -> Dict[str, Any]:
    prompt = messages_to_prompt(req.messages)
    inputs: Dict[str, Any] = {settings.airops_input_key: prompt}
    if settings.include_raw_messages:
        inputs["messages"] = [m.model_dump() for m in req.messages]
    return {"inputs": inputs}


def extract_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        for item in value:
            found = extract_text(item)
            if found:
                return found
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, dict):
        for key in (
            "output",
            "result",
            "text",
            "content",
            "message",
            "answer",
            "response",
            "body",
        ):
            if key in value:
                found = extract_text(value[key])
                if found:
                    return found
        for key in ("outputs", "data", "run", "execution"):
            if key in value:
                found = extract_text(value[key])
                if found:
                    return found
        return json.dumps(value, ensure_ascii=False)
    return str(value)


async def call_airops(req: ChatCompletionRequest) -> str:
    require_airops_config()
    payload = build_airops_payload(req)
    headers = {
        "Authorization": f"Bearer {settings.airops_api_key}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=settings.request_timeout) as client:
        response = await client.post(
            settings.airops_execute_url,
            headers=headers,
            json=payload,
        )
    if response.status_code >= 400:
        body = response.text[:2000]
        raise HTTPException(
            status_code=502,
            detail=f"AirOps returned {response.status_code}: {body}",
        )
    try:
        data = response.json()
    except ValueError:
        return response.text
    return extract_text(data)


def completion_object(answer: str, model: str) -> Dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


def stream_chunks(answer: str, model: str):
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    first = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(first, ensure_ascii=False)}\n\n"
    body = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"content": answer}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(body, ensure_ascii=False)}\n\n"
    end = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(end, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


@app.get("/")
async def root() -> Dict[str, Any]:
    return {
        "ok": True,
        "service": "AirOps SillyTavern Proxy",
        "model": settings.exposed_model,
        "endpoints": ["/v1/models", "/v1/chat/completions"],
    }


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "has_airops_execute_url": bool(settings.airops_execute_url),
        "has_airops_api_key": bool(settings.airops_api_key),
        "proxy_auth_enabled": bool(settings.proxy_api_key),
    }


@app.get("/v1/models")
async def models(authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    require_proxy_auth(authorization)
    return {
        "object": "list",
        "data": [
            {
                "id": settings.exposed_model,
                "object": "model",
                "created": 0,
                "owned_by": "airops",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(
    req: ChatCompletionRequest,
    authorization: Optional[str] = Header(default=None),
):
    require_proxy_auth(authorization)
    answer = await call_airops(req)
    model = req.model or settings.exposed_model
    if req.stream:
        return StreamingResponse(
            stream_chunks(answer, model),
            media_type="text/event-stream",
        )
    return JSONResponse(completion_object(answer, model))
