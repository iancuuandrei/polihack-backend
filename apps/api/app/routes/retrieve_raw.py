from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends

from ..db.session import session_context
from ..schemas import RawRetrievalRequest, RawRetrievalResponse
from ..services.raw_retriever import (
    EmptyRawRetrievalStore,
    PostgresRawRetrievalStore,
    RawRetriever,
)


router = APIRouter(tags=["retrieval"])


async def get_raw_retriever() -> AsyncIterator[RawRetriever]:
    try:
        async with session_context() as session:
            yield RawRetriever(PostgresRawRetrievalStore(session))
    except RuntimeError as exc:
        yield RawRetriever(
            EmptyRawRetrievalStore(),
            initial_warnings=[f"database_unavailable:{type(exc).__name__}"],
        )


@router.post("/retrieve/raw", response_model=RawRetrievalResponse)
async def retrieve_raw(
    request: RawRetrievalRequest,
    retriever: RawRetriever = Depends(get_raw_retriever),
) -> RawRetrievalResponse:
    return await retriever.retrieve(request)
