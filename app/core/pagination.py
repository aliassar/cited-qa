"""
Cursor (keyset) pagination.

WHY NOT OFFSET/LIMIT
--------------------
`ORDER BY created_at DESC OFFSET 100000 LIMIT 20` forces the database to walk
and discard 100,000 rows on every page. Cost grows linearly with page depth.
Worse, it is *unstable*: if a row is inserted while the user pages, items shift
and they see duplicates or skip records.

KEYSET PAGINATION
-----------------
Instead of "skip N", we say "give me rows strictly after this exact position".
The position is the sort key of the last row you saw. With an index on that key
the database seeks straight to it - O(log n), constant cost per page, and no
duplicates or gaps under concurrent inserts.

    WHERE (created_at, id) < (:last_created_at, :last_id)
    ORDER BY created_at DESC, id DESC
    LIMIT :page_size

THE TIE-BREAKER MATTERS. `created_at` alone is not unique: two documents
created in the same millisecond would make the comparison ambiguous and you
would silently lose rows. Appending the primary key makes the sort key total.

TRADE-OFF: you cannot jump to "page 47",
and you cannot show a total page count cheaply. That is almost always fine for
feeds and infinite scroll; it is not fine for a numbered pager over a small
table, where OFFSET is genuinely the right tool.

The cursor is base64(JSON) - opaque to the client so we can change the internal
shape without breaking them. It is NOT encrypted: never put secrets in it.
"""

from __future__ import annotations

import base64
import binascii
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from app.core.errors import ValidationError


@dataclass(frozen=True)
class Cursor:
    """Decoded position: the sort key of the last row of the previous page."""

    created_at: datetime
    id: str

    def encode(self) -> str:
        raw = json.dumps({"c": self.created_at.isoformat(), "i": self.id})
        # urlsafe_b64 so the cursor can live in a query string untouched.
        # rstrip("=") drops padding (also query-string friendly); we add it back on decode.
        return base64.urlsafe_b64encode(raw.encode()).decode().rstrip("=")

    @classmethod
    def decode(cls, token: str) -> Cursor:
        try:
            padding = "=" * (-len(token) % 4)
            raw = base64.urlsafe_b64decode(token + padding)
            data: dict[str, Any] = json.loads(raw)
            return cls(created_at=datetime.fromisoformat(data["c"]), id=data["i"])
        except (binascii.Error, ValueError, KeyError, TypeError):
            # Never 500 on user input. A malformed cursor is a client error.
            raise ValidationError("Malformed pagination cursor") from None


@dataclass
class Page:
    """A slice of results plus the cursor for the next slice."""

    items: list[Any]
    next_cursor: str | None

    @property
    def has_more(self) -> bool:
        return self.next_cursor is not None


def build_page(rows: list[Any], limit: int) -> Page:
    """
    Turn `limit + 1` fetched rows into a page.

    THE +1 TRICK: query for one more row than the client asked for. If it comes
    back, there is at least one more page - and you learn that without a second
    COUNT(*) query. Drop the extra row before returning.
    """
    has_more = len(rows) > limit
    items = rows[:limit]
    next_cursor = (
        Cursor(created_at=items[-1].created_at, id=items[-1].id).encode()
        if has_more and items
        else None
    )
    return Page(items=items, next_cursor=next_cursor)
