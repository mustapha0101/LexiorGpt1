# -*- coding: utf-8 -*-
"""Checkpointing du graphe central.

Un checkpointer est REQUIS pour ``interrupt()`` (clarifications live).
Mémoire par défaut (threads de conversation en cours de processus);
SQLite pour la persistance locale; PostgreSQL en production.
"""

from __future__ import annotations


def create_memory_checkpointer():
    """Checkpointer en mémoire — défaut de l'API live."""
    from langgraph.checkpoint.memory import MemorySaver
    return MemorySaver()


def create_sqlite_checkpointer(db_path: str = ":memory:"):
    """Create a SQLite-backed checkpointer for local use."""
    from langgraph.checkpoint.sqlite import SqliteSaver
    return SqliteSaver.from_conn_string(db_path)


def create_async_sqlite_checkpointer(db_path: str = ":memory:"):
    """Create an async SQLite checkpointer for use with FastAPI."""
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    return AsyncSqliteSaver.from_conn_string(db_path)
