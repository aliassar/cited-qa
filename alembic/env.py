"""
Alembic environment.

WHY MIGRATIONS INSTEAD OF Base.metadata.create_all()
-----------------------------------------------------
create_all only ever CREATES missing tables. It cannot rename a column, change
a type, backfill data, or add an index concurrently - and it has no notion of
order or of rollback. The moment you have a production database with real rows,
you need an ordered, reviewable, reversible list of schema changes. That is
Alembic.

Two rules that matter operationally:
  * Migrations run as a separate step BEFORE the new code is deployed, never on
    application startup. Startup migrations mean N pods racing to run the same
    DDL.
  * Every migration must be backward compatible with the currently running
    code, because during a rolling deploy both versions are live at once. Adding
    a NOT NULL column with no default breaks the old pods instantly. The safe
    pattern is expand -> migrate -> contract: add nullable, backfill, start
    writing, then tighten in a later release.
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context
from app.core.config import get_settings
from app.db import models  # noqa: F401  (imported for its side effect)

# Importing the models is what populates Base.metadata, which is what
# --autogenerate diffs against. Forget this import and Alembic will cheerfully
# generate a migration that DROPS every table.
from app.db.base import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# One source of truth for the URL: the application settings.
config.set_main_option("sqlalchemy.url", get_settings().database_url)

target_metadata = Base.metadata


def _configure(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        # Detect column TYPE changes too - off by default, and its absence is a
        # common source of "the migration ran but the column is still VARCHAR".
        compare_type=True,
        compare_server_default=True,
        # Batch mode: SQLite cannot ALTER most things, so Alembic emulates it by
        # creating a new table, copying rows and swapping. Harmless on Postgres.
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_offline() -> None:
    """Emit SQL to stdout instead of executing it (`alembic upgrade head --sql`).

    This is how you hand a DBA a reviewable script for a production change."""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        # NullPool: a migration is a one-shot process. Pooling would only keep
        # connections open after the work is done.
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(_configure)
    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
