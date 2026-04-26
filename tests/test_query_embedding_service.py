import httpx
import pytest

from apps.api.app.services.query_embedding_service import (
    QUERY_EMBEDDING_NOT_CONFIGURED,
    QUERY_EMBEDDING_UNAVAILABLE,
    QueryEmbeddingService,
)


@pytest.mark.anyio
async def test_query_embedding_service_disabled_returns_no_embedding():
    result = await QueryEmbeddingService(
        enabled=False,
        base_url="http://127.0.0.1:11434",
        model="fixture-embedding",
    ).embed("Poate angajatorul sa-mi scada salariul?", debug=True)

    assert result.embedding is None
    assert result.enabled is False
    assert result.available is False
    assert result.warnings == []
    assert result.debug["enabled"] is False


@pytest.mark.anyio
async def test_query_embedding_service_missing_base_url_warns_not_configured():
    result = await QueryEmbeddingService(
        enabled=True,
        base_url="",
        model="fixture-embedding",
    ).embed("Poate angajatorul sa-mi scada salariul?", debug=True)

    assert result.embedding is None
    assert result.enabled is True
    assert result.available is False
    assert QUERY_EMBEDDING_NOT_CONFIGURED in result.warnings
    assert result.debug["fallback_reason"] == "not_configured"


@pytest.mark.anyio
async def test_query_embedding_service_parses_ollama_embeddings_response():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/embed"
        payload = request.content.decode("utf-8")
        assert "fixture-embedding" in payload
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2]]})

    result = await QueryEmbeddingService(
        enabled=True,
        base_url="http://ollama.local",
        model="fixture-embedding",
        transport=httpx.MockTransport(handler),
    ).embed("Poate angajatorul sa-mi scada salariul?", debug=True)

    assert result.embedding == [0.1, 0.2]
    assert result.model == "fixture-embedding"
    assert result.dimension == 2
    assert result.available is True
    assert result.warnings == []
    assert result.debug["dimension"] == 2


@pytest.mark.anyio
async def test_query_embedding_service_parses_single_embedding_response():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/embed"
        return httpx.Response(200, json={"embedding": [0.1, 0.2]})

    result = await QueryEmbeddingService(
        enabled=True,
        base_url="http://ollama.local",
        model="fixture-embedding",
        transport=httpx.MockTransport(handler),
    ).embed("Poate angajatorul sa-mi scada salariul?", debug=True)

    assert result.embedding == [0.1, 0.2]
    assert result.dimension == 2
    assert result.available is True


@pytest.mark.anyio
async def test_query_embedding_service_falls_back_to_legacy_embeddings_endpoint():
    requested_paths = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)
        if request.url.path == "/api/embed":
            return httpx.Response(404, json={"error": "not found"})
        return httpx.Response(200, json={"embedding": [0.1, 0.2]})

    result = await QueryEmbeddingService(
        enabled=True,
        base_url="http://ollama.local",
        model="fixture-embedding",
        transport=httpx.MockTransport(handler),
    ).embed("Poate angajatorul sa-mi scada salariul?", debug=True)

    assert requested_paths == ["/api/embed", "/api/embeddings"]
    assert result.embedding == [0.1, 0.2]
    assert result.dimension == 2
    assert result.available is True


@pytest.mark.anyio
async def test_query_embedding_service_invalid_response_warns_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": []})

    result = await QueryEmbeddingService(
        enabled=True,
        base_url="http://ollama.local",
        model="fixture-embedding",
        transport=httpx.MockTransport(handler),
    ).embed("Poate angajatorul sa-mi scada salariul?", debug=True)

    assert result.embedding is None
    assert result.available is False
    assert QUERY_EMBEDDING_UNAVAILABLE in result.warnings
    assert result.debug["fallback_reason"] == "invalid_response"
