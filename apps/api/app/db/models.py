from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import Date, DateTime, Float, JSON, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class LegalUnit(Base):
    __tablename__ = "legal_units"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    canonical_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    law_id: Mapped[str] = mapped_column(Text, nullable=False)
    law_title: Mapped[str] = mapped_column(Text, nullable=False)
    act_type: Mapped[str | None] = mapped_column(Text, nullable=True)
    act_number: Mapped[str | None] = mapped_column(Text, nullable=True)
    publication_date: Mapped[Any | None] = mapped_column(Date, nullable=True)
    effective_date: Mapped[Any | None] = mapped_column(Date, nullable=True)
    version_start: Mapped[Any | None] = mapped_column(Date, nullable=True)
    version_end: Mapped[Any | None] = mapped_column(Date, nullable=True)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="unknown")
    hierarchy_path: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    article_number: Mapped[str | None] = mapped_column(Text, nullable=True)
    paragraph_number: Mapped[str | None] = mapped_column(Text, nullable=True)
    letter_number: Mapped[str | None] = mapped_column(Text, nullable=True)
    point_number: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    legal_domain: Mapped[str | None] = mapped_column(Text, nullable=True)
    legal_concepts: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    parent_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    parser_warnings: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class LegalEdge(Base):
    __tablename__ = "legal_edges"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    source_id: Mapped[str] = mapped_column(Text, nullable=False)
    target_id: Mapped[str] = mapped_column(Text, nullable=False)
    type: Mapped[str] = mapped_column(Text, nullable=False)
    weight: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    metadata_: Mapped[Any | None] = mapped_column("metadata", JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class ImportRun(Base):
    __tablename__ = "import_runs"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    source_dir: Mapped[str | None] = mapped_column(Text, nullable=True)
    mode: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str | None] = mapped_column(Text, nullable=True)
    counts: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    warnings: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    errors: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
