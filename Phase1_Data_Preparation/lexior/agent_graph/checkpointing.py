# -*- coding: utf-8 -*-
"""Checkpoint configuration for the Lexior agent graph.

SQLite for local development; PostgreSQL for production (Phase 2+).
"""

from __future__ import annotations

from typing import Optional


def create_sqlite_checkpointer(db_path: str = ":memory:"):
    """Create a SQLite-backed checkpointer for local use."""
    from langgraph.checkpoint.sqlite import SqliteSaver
    return SqliteSaver.from_conn_string(db_path)


def create_async_sqlite_checkpointer(db_path: str = ":memory:"):
    """Create an async SQLite checkpointer for use with FastAPI."""
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    return AsyncSqliteSaver.from_conn_string(db_path)
