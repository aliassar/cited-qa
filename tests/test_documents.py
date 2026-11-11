"""Upload, async ingestion, content dedupe, optimistic locking, permissions."""

from __future__ import annotations

from httpx import AsyncClient
from sqlalchemy import func, select

from app.db.models import DocumentChunk, DocumentStatus
from app.services import documents as document_service
from app.services.ingestion import ingest_now
from tests.conftest import SAMPLE_CONTRACT, register_and_login, upload_document, wait_for_status


async def _new_matter(client: AsyncClient, auth: dict) -> str:
    response = await client.post(
        "/api/v1/matters", headers=auth["headers"], json={"title": "Ingestion Matter"}
    )
    return response.json()["id"]


async def test_upload_returns_202_and_ingests_in_the_background(client: AsyncClient) -> None:
    """
    The upload responds 202 with status=pending, and the document becomes
    'ready' shortly afterwards - without the client having waited for it.
    """
    auth = await register_and_login(client, org="Ingest Firm", email="ing@example.com")
    matter_id = await _new_matter(client, auth)

    response = await client.post(
        f"/api/v1/matters/{matter_id}/documents",
        headers=auth["headers"],
        files={"file": ("nda.txt", SAMPLE_CONTRACT.encode(), "text/plain")},
        data={"title": "Mutual NDA"},
    )
    assert response.status_code == 202
    # The Location header is what makes 202 actionable: it says where to poll.
    assert response.headers["Location"].endswith(response.json()["id"])
    assert response.json()["status"] == "pending"

    document = await wait_for_status(client, auth["headers"], response.json()["id"])
    assert document["status"] == "ready"
    # The sample contract is short enough to pack into a single chunk - which is
    # correct behaviour, not a bug. Chunk-boundary logic is covered directly in
    # tests/test_chunking.py, where we can control the input size precisely.
    assert document["chunk_count"] >= 1


async def test_uploading_the_same_bytes_twice_is_deduplicated(client: AsyncClient) -> None:
    """Content-addressed by SHA-256, enforced by a unique constraint."""
    auth = await register_and_login(client, org="Dedupe Firm", email="dd@example.com")
    matter_id = await _new_matter(client, auth)

    first = await upload_document(client, auth["headers"], matter_id)
    second = await client.post(
        f"/api/v1/matters/{matter_id}/documents",
        headers=auth["headers"],
        files={"file": ("nda-copy.txt", SAMPLE_CONTRACT.encode(), "text/plain")},
    )
    # 200, not 202: nothing new was accepted for processing.
    assert second.status_code == 200
    assert second.json()["id"] == first["id"]


async def test_unsupported_content_type_is_415(client: AsyncClient) -> None:
    auth = await register_and_login(client, org="Type Firm", email="ty@example.com")
    matter_id = await _new_matter(client, auth)

    response = await client.post(
        f"/api/v1/matters/{matter_id}/documents",
        headers=auth["headers"],
        files={"file": ("virus.exe", b"MZ\x90\x00", "application/x-msdownload")},
    )
    assert response.status_code == 415


async def test_oversized_upload_is_413(client: AsyncClient, settings) -> None:
    auth = await register_and_login(client, org="Big Firm", email="big@example.com")
    matter_id = await _new_matter(client, auth)

    payload = b"x" * (settings.max_upload_bytes + 1)
    response = await client.post(
        f"/api/v1/matters/{matter_id}/documents",
        headers=auth["headers"],
        files={"file": ("huge.txt", payload, "text/plain")},
    )
    assert response.status_code == 413


async def test_pdf_upload_is_415(client: AsyncClient) -> None:
    """There is no PDF extractor, so PDFs are not on the upload allow-list."""
    auth = await register_and_login(client, org="Pdf Firm", email="pdf@example.com")
    matter_id = await _new_matter(client, auth)

    response = await client.post(
        f"/api/v1/matters/{matter_id}/documents",
        headers=auth["headers"],
        files={"file": ("contract.pdf", b"%PDF-1.7 ...", "application/pdf")},
    )
    assert response.status_code == 415


async def test_empty_upload_is_422(client: AsyncClient) -> None:
    """An empty file is invalid content, not an oversized one."""
    auth = await register_and_login(client, org="Empty Firm", email="empty@example.com")
    matter_id = await _new_matter(client, auth)

    response = await client.post(
        f"/api/v1/matters/{matter_id}/documents",
        headers=auth["headers"],
        files={"file": ("empty.txt", b"", "text/plain")},
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "Uploaded file is empty"


async def test_ingest_now_matches_the_worker_and_is_idempotent(client: AsyncClient) -> None:
    """
    `ingest_now` (used by the seed script) runs the same `ingest_document` as
    the background worker: a re-run converges to the same chunks instead of
    duplicating them, and a document with no text ends up FAILED.
    """
    auth = await register_and_login(client, org="Inline Firm", email="inline@example.com")
    matter_id = await _new_matter(client, auth)
    org_id = auth["user"]["org_id"]

    session_factory = client.app.state.session_factory  # type: ignore[attr-defined]
    async with session_factory() as session:
        document, created = await document_service.create_document(
            session,
            org_id=org_id,
            matter_id=matter_id,
            title="Inline",
            content_type="text/plain",
            payload=SAMPLE_CONTRACT.encode(),
        )
        assert created
        await ingest_now(session, document.id)
        await ingest_now(session, document.id)  # a redelivery
        await session.refresh(document)
        assert document.status == DocumentStatus.READY

        stored = await session.scalar(
            select(func.count())
            .select_from(DocumentChunk)
            .where(DocumentChunk.document_id == document.id)
        )
        assert stored == document.chunk_count >= 1

        blank, _ = await document_service.create_document(
            session,
            org_id=org_id,
            matter_id=matter_id,
            title="Blank",
            content_type="text/plain",
            payload=b"   \n\n  ",
        )
        await ingest_now(session, blank.id)
        await session.refresh(blank)
        assert blank.status == DocumentStatus.FAILED
        assert blank.failure_reason == "No extractable text found in document"


async def test_optimistic_locking_rejects_a_stale_write(client: AsyncClient) -> None:
    """
    THE CONCURRENCY TEST.

    Two users read version 1. The first update succeeds and bumps it to 2. The
    second still holds version 1, so its If-Match no longer matches and the
    write is refused with 409 - instead of silently overwriting the first
    user's change ("last write wins" is data loss, not a policy).
    """
    auth = await register_and_login(client, org="Lock Firm", email="lock@example.com")
    matter_id = await _new_matter(client, auth)
    document = await upload_document(client, auth["headers"], matter_id)

    read = await client.get(f"/api/v1/documents/{document['id']}", headers=auth["headers"])
    version = read.json()["version"]
    assert read.headers["ETag"] == f'"{version}"'  # ETag is how the client learns it

    winner = await client.patch(
        f"/api/v1/documents/{document['id']}",
        headers={**auth["headers"], "If-Match": str(version)},
        json={"title": "Renamed by user A"},
    )
    assert winner.status_code == 200
    assert winner.json()["version"] == version + 1

    loser = await client.patch(
        f"/api/v1/documents/{document['id']}",
        headers={**auth["headers"], "If-Match": str(version)},  # stale
        json={"title": "Renamed by user B"},
    )
    assert loser.status_code == 409
    assert loser.json()["current_version"] == version + 1

    final = await client.get(f"/api/v1/documents/{document['id']}", headers=auth["headers"])
    assert final.json()["title"] == "Renamed by user A"  # A's change survived


async def test_update_without_if_match_still_works(client: AsyncClient) -> None:
    """
    If-Match is opt-in here: omitting it means "I accept last-write-wins".

    A stricter API would REQUIRE If-Match on every mutating request and return
    428 Precondition Required without it. That is the right call for genuinely
    collaborative documents; it is a product decision, not a technical one.
    """
    auth = await register_and_login(client, org="NoMatch Firm", email="nm@example.com")
    matter_id = await _new_matter(client, auth)
    document = await upload_document(client, auth["headers"], matter_id)

    response = await client.patch(
        f"/api/v1/documents/{document['id']}",
        headers=auth["headers"],
        json={"title": "Unconditional rename"},
    )
    assert response.status_code == 200


async def test_documents_are_tenant_scoped(client: AsyncClient) -> None:
    firm_a = await register_and_login(client, org="Doc Firm A", email="da@example.com")
    firm_b = await register_and_login(client, org="Doc Firm B", email="db@example.com")

    matter_id = await _new_matter(client, firm_a)
    document = await upload_document(client, firm_a["headers"], matter_id)

    assert (
        await client.get(f"/api/v1/documents/{document['id']}", headers=firm_b["headers"])
    ).status_code == 404


async def test_status_filter_on_the_list_endpoint(client: AsyncClient) -> None:
    auth = await register_and_login(client, org="Filter Firm", email="fi@example.com")
    matter_id = await _new_matter(client, auth)
    await upload_document(client, auth["headers"], matter_id)

    ready = await client.get(
        f"/api/v1/matters/{matter_id}/documents?status=ready", headers=auth["headers"]
    )
    failed = await client.get(
        f"/api/v1/matters/{matter_id}/documents?status=failed", headers=auth["headers"]
    )
    assert len(ready.json()["items"]) == 1
    assert failed.json()["items"] == []
