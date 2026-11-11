"""
Async engine, session factory, and the FastAPI session dependency.

THE ASYNC STORY
---------------
FastAPI runs an event loop. `async def` handlers execute ON that loop. If you
call a blocking function inside one - `requests.get`, `time.sleep`, psycopg2,
bcrypt - you block the loop, and EVERY concurrent request on that worker stalls.
Async is not "faster code"; it is "the CPU does something else while we wait on
the network". If you never await I/O, async buys you nothing.

So: async driver (asyncpg / aiosqlite), async engine, async session, and any
genuinely CPU-bound work pushed to a thread pool.

FastAPI's escape hatch: a plain `def` handler is automatically run in a
threadpool, so legacy blocking code is safe there. The bug is the hybrid -
`async def` containing blocking calls. Nothing warns you; throughput just dies
under load.

SESSION LIFECYCLE
-----------------
One session per request, opened by a dependency and closed when the response is
done. Sessions are NOT thread-safe or task-safe: never share one across
concurrent tasks, and never make one a module-level global.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import Request
from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool, StaticPool

from app.core.config import Settings


def create_engine(settings: Settings) -> AsyncEngine:
    kwargs: dict = {
        "echo": settings.db_echo,
        # pool_pre_ping issues a cheap "SELECT 1" before handing out a pooled
        # connection. Without it, connections killed by a proxy/firewall idle
        # timeout (or a DB failover) surface as random "server closed the
        # connection unexpectedly" errors on the NEXT request. Cheap insurance.
        "pool_pre_ping": True,
    }

    if settings.is_sqlite:
        # check_same_thread=False because aiosqlite drives the sqlite3 module
        # from a worker thread that differs from the calling thread.
        kwargs["connect_args"] = {"check_same_thread": False}

        if ":memory:" in settings.database_url:
            # An in-memory database lives inside ONE connection and disappears
            # when it closes, so we must pin a single shared connection.
            #
            # THE CATCH, and it bit this project during development: with one
            # shared connection every session interleaves on the same
            # transaction. A request finishing and closing its session issues a
            # ROLLBACK that discards a background worker's uncommitted writes,
            # and the document sits in "processing" forever. It looks exactly
            # like a concurrency bug in your code, and it is not - it is the
            # test database lying about isolation.
            #
            # So: in-memory is fine for single-connection tests, and wrong the
            # moment a background task writes concurrently. Use a temp FILE for
            # those (see tests/conftest.py).
            kwargs["poolclass"] = StaticPool
        else:
            # File-backed SQLite: give every session its OWN connection so
            # transactions are genuinely isolated, like Postgres.
            kwargs["poolclass"] = NullPool
    else:
        kwargs.update(
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_timeout=settings.db_pool_timeout_seconds,
            # Recycle connections before the DB's own idle timeout closes them.
            pool_recycle=1800,
        )

    engine = create_async_engine(settings.database_url, **kwargs)

    if settings.is_sqlite:

        @event.listens_for(engine.sync_engine, "connect")
        def _sqlite_pragmas(dbapi_connection, _record):  # pragma: no cover - trivial
            cursor = dbapi_connection.cursor()
            # SQLite does NOT enforce foreign keys unless you ask, per
            # connection. Without this, ON DELETE CASCADE silently does nothing
            # and your tests pass while Postgres behaves differently.
            cursor.execute("PRAGMA foreign_keys=ON")
            if ":memory:" not in settings.database_url:
                # WAL lets readers and one writer proceed concurrently instead
                # of blocking each other; busy_timeout makes a writer WAIT for a
                # lock rather than failing instantly with "database is locked".
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA busy_timeout=5000")
            cursor.close()

    return engine


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        # expire_on_commit=False is important in async code. The default (True)
        # marks every attribute stale after commit, so touching `obj.id`
        # afterwards triggers a lazy refresh - which in async SQLAlchemy raises
        # MissingGreenlet, because implicit IO cannot happen outside an await.
        # Turning it off lets us commit and then serialise the object.
        expire_on_commit=False,
        autoflush=False,  # be explicit about when SQL is emitted
    )


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    """
    FastAPI dependency yielding one AsyncSession per request.

    NOTE the `request: Request` ANNOTATION. FastAPI inspects the type hint to
    decide what a parameter is; leave it untyped and FastAPI assumes it is a
    query parameter named "request" and every endpoint 422s with
    "request: Field required". Annotate your dependency parameters.

    The `try/finally` shape matters. FastAPI runs the code after `yield` once
    the response has been produced, so this is where cleanup belongs.

    We deliberately do NOT auto-commit here. The service functions (and the
    idempotency guard in api/deps.py) commit explicitly, so "what got persisted,
    and when" is visible in the unit of work itself rather than hidden in a
    dependency. Auto-commit-on-success is a defensible alternative; pick one and
    be consistent.
    """
    factory: async_sessionmaker[AsyncSession] = request.app.state.session_factory
    session = factory()
    try:
        yield session
    except Exception:
        # An unhandled error means the unit of work failed: undo partial writes.
        await session.rollback()
        raise
    finally:
        # Return the connection to the pool. Leaking sessions exhausts the pool
        # and the app hangs waiting for one - a classic production outage.
        await session.close()
