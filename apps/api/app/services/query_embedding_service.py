from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

from ..config import settings


QUERY_EMBEDDING_NOT_CONFIGURED = "query_embedding_not_configured"
QUERY_EMBEDDING_UNAVAILABLE = "query_embedding_unavailable"


@dataclass
class QueryEmbeddingResult:
    embedding: list[float] | None = None
    model: str | None = None
    dimension: int | None = None
    enabled: bool = False
    available: bool = False
    warnings: list[str] = field(default_factory=list)
    debug: dict[str, Any] | None = None


class QueryEmbeddingService:
    def __init__(
        self,
        *,
        enabled: bool | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._enabled = enabled
        self._base_url = base_url
        self._model = model
        self._timeout_seconds = timeout_seconds
        self.transport = transport

    @property
    def enabled(self) -> bool:
        if self._enabled is not None:
            return self._enabled
        return bool(settings.query_embedding_enabled)

    @property
    def base_url(self) -> str | None:
        return self._base_url if self._base_url is not None else settings.ollama_base_url

    @property
    def model(self) -> str | None:
        return self._model if self._model is not None else settings.query_embedding_model

    @property
    def timeout_seconds(self) -> float:
        if self._timeout_seconds is not None:
            return self._timeout_seconds
        return float(settings.query_embedding_timeout_seconds or 20.0)

    async def embed(
        self,
        question: str,
        *,
        debug: bool = False,
    ) -> QueryEmbeddingResult:
        debug_payload = self._base_debug() if debug else None
        if not self.enabled:
            return QueryEmbeddingResult(
                enabled=False,
                available=False,
                model=self.model,
                debug=debug_payload,
            )

        if not self.base_url or not self.model:
            if debug_payload is not None:
                debug_payload["fallback_reason"] = "not_configured"
            return QueryEmbeddingResult(
                enabled=True,
                available=False,
                model=self.model,
                warnings=[QUERY_EMBEDDING_NOT_CONFIGURED],
                debug=debug_payload,
            )

        if debug_payload is not None:
            debug_payload["attempted"] = True
        started_at = time.perf_counter()
        try:
            payload = await self._post_embed(
                {
                    "model": self.model,
                    "input": question,
                }
            )
            embedding = _extract_embedding(payload)
        except Exception as exc:
            if debug_payload is not None:
                debug_payload["fallback_reason"] = _fallback_reason(exc)
                debug_payload["latency_ms"] = _elapsed_ms(started_at)
            return QueryEmbeddingResult(
                enabled=True,
                available=False,
                model=self.model,
                warnings=[QUERY_EMBEDDING_UNAVAILABLE],
                debug=debug_payload,
            )

        if debug_payload is not None:
            debug_payload["available"] = True
            debug_payload["dimension"] = len(embedding)
            debug_payload["latency_ms"] = _elapsed_ms(started_at)
        return QueryEmbeddingResult(
            embedding=embedding,
            model=self.model,
            dimension=len(embedding),
            enabled=True,
            available=True,
            warnings=[],
            debug=debug_payload,
        )

    async def _post_embed(self, payload: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(
            timeout=self.timeout_seconds,
            transport=self.transport,
        ) as client:
            response = await client.post(
                self._endpoint_url(),
                json=payload,
            )
            if response.status_code in {404, 405}:
                response = await client.post(
                    self._legacy_endpoint_url(),
                    json=payload,
                )
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise ValueError("embedding_response_must_be_object")
            return data

    def _endpoint_url(self) -> str:
        base_url = (self.base_url or "").strip().rstrip("/")
        if not base_url:
            raise RuntimeError("query embedding endpoint is not configured")
        if base_url.endswith("/api/embed"):
            return base_url
        return f"{base_url}/api/embed"

    def _legacy_endpoint_url(self) -> str:
        base_url = (self.base_url or "").strip().rstrip("/")
        if not base_url:
            raise RuntimeError("query embedding endpoint is not configured")
        if base_url.endswith("/api/embed"):
            return f"{base_url[: -len('/api/embed')]}/api/embeddings"
        if base_url.endswith("/api/embeddings"):
            return base_url
        return f"{base_url}/api/embeddings"

    def _base_debug(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "attempted": False,
            "available": False,
            "model": self.model,
            "dimension": None,
            "fallback_reason": None,
            "latency_ms": None,
        }


def _extract_embedding(payload: Mapping[str, Any]) -> list[float]:
    direct_embedding = payload.get("embedding")
    if isinstance(direct_embedding, Sequence) and not isinstance(
        direct_embedding,
        (str, bytes, bytearray),
    ):
        return _coerce_embedding_vector(direct_embedding)

    embeddings = payload.get("embeddings")
    if isinstance(embeddings, Sequence) and not isinstance(
        embeddings,
        (str, bytes, bytearray),
    ):
        if embeddings and isinstance(embeddings[0], Sequence) and not isinstance(
            embeddings[0],
            (str, bytes, bytearray),
        ):
            return _coerce_embedding_vector(embeddings[0])
        return _coerce_embedding_vector(embeddings)

    raise ValueError("missing_embedding")


def _coerce_embedding_vector(values: Sequence[Any]) -> list[float]:
    vector: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("invalid_embedding_value")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("invalid_embedding_value")
        vector.append(number)
    if not vector:
        raise ValueError("empty_embedding")
    return vector


def _fallback_reason(exc: Exception) -> str:
    if isinstance(exc, httpx.TimeoutException):
        return "timeout"
    if isinstance(exc, httpx.HTTPError):
        return "http_error"
    return "invalid_response"


def _elapsed_ms(started_at: float) -> int:
    return round((time.perf_counter() - started_at) * 1000)
