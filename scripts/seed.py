"""
Seed the database with demo data.

    python -m scripts.seed

Creates one organisation, one admin user, one matter and three short contracts,
ingesting them synchronously so everything is immediately queryable. Useful for
poking at the API by hand and for demos.

Idempotent: running it again finds the existing demo user and matter and
creates nothing new (documents are deduplicated by checksum anyway).

Note the structure: this script reuses the SAME service functions the HTTP
endpoints call. That is the payoff of keeping business logic out of the
handlers - a CLI, a background job and an endpoint can all drive it.
"""

from __future__ import annotations

import asyncio

from sqlalchemy import select

from app.core.config import get_settings
from app.db.base import Base
from app.db.models import Matter, User
from app.db.session import create_engine, create_session_factory
from app.services.auth import register_organization
from app.services.documents import create_document
from app.services.ingestion import ingest_now
from app.services.matters import create_matter

DEMO_EMAIL = "demo@example.com"
DEMO_PASSWORD = "demo-password-please-change"
DEMO_ORG_NAME = "Demo Legal Partners"
DEMO_MATTER_REFERENCE = "GLOBEX-2026-004"

CONTRACTS: dict[str, str] = {
    "Mutual NDA.txt": """\
1. Definitions

"Confidential Information" means any information disclosed by one party to the
other, whether orally, in writing or in any other form, that is designated as
confidential or that would reasonably be understood to be confidential.

2. Obligations

The Receiving Party shall not disclose Confidential Information to any third
party without the prior written consent of the Disclosing Party, save to its
professional advisers who are bound by equivalent obligations of confidence.

3. Term

The obligations in this Agreement survive termination and continue for five (5)
years from the date of disclosure.

4. Governing Law

This Agreement is governed by the laws of the Federal Republic of Germany.
""",
    "Master Services Agreement.txt": """\
1. Services and Fees

The Supplier shall provide the Services described in each Statement of Work.
Fees are payable within thirty (30) days of a valid invoice. Late payment
accrues interest at 8% above the base rate.

2. Term and Termination

This Agreement commences on the Effective Date and continues for twenty-four
(24) months. Either party may terminate for convenience on ninety (90) days
written notice, or immediately on the other party's material breach that remains
unremedied for thirty (30) days.

3. Limitation of Liability

Neither party's aggregate liability shall exceed the fees paid in the twelve
months preceding the claim. Nothing limits liability for death, personal injury,
or fraud.

4. Data Protection

Each party shall comply with the General Data Protection Regulation. The
Supplier acts as processor and shall not transfer personal data outside the
European Economic Area without an adequate transfer mechanism.
""",
    "Employment Contract.txt": """\
1. Position and Duties

The Employee is engaged as Operations Manager and shall report to the
Managing Director.

2. Probation and Notice

The first six (6) months constitute a probationary period during which either
party may terminate on two (2) weeks notice. Thereafter the statutory notice
period under section 622 BGB applies.

3. Working Time

The regular working week is forty (40) hours. Overtime is compensated by time
off in lieu, to be taken within three months.

4. Post-Contractual Non-Compete

Any non-compete obligation after termination requires compensation of at least
50% of the Employee's most recent remuneration, in accordance with sections
74 ff. HGB.
""",
}


async def main() -> None:
    settings = get_settings()
    engine = create_engine(settings)
    session_factory = create_session_factory(engine)

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    async with session_factory() as session:
        # users.email is globally unique, so this matches at most one row.
        existing = await session.scalar(select(User).where(User.email == DEMO_EMAIL))
        if existing is not None:
            print(f"Demo user already exists: {DEMO_EMAIL}")
            user = existing
        else:
            user = await register_organization(
                session,
                settings,
                org_name=DEMO_ORG_NAME,
                email=DEMO_EMAIL,
                full_name="Demo Lawyer",
                password=DEMO_PASSWORD,
            )
            print(f"Created organisation '{DEMO_ORG_NAME}' and admin user {DEMO_EMAIL}")

        matter = await session.scalar(
            select(Matter).where(
                Matter.org_id == user.org_id,
                Matter.client_reference == DEMO_MATTER_REFERENCE,
            )
        )
        if matter is not None:
            print(f"Demo matter already exists: {matter.id}: {matter.title}")
        else:
            matter = await create_matter(
                session,
                org_id=user.org_id,
                user_id=user.id,
                title="Globex onboarding - contract review",
                client_reference=DEMO_MATTER_REFERENCE,
            )
            print(f"Created matter {matter.id}: {matter.title}")

        for name, body in CONTRACTS.items():
            document, created = await create_document(
                session,
                org_id=user.org_id,
                matter_id=matter.id,
                title=name,
                content_type="text/plain",
                payload=body.encode(),
            )
            if created:
                # Ingest inline rather than through the queue: a one-shot script
                # would exit before background workers finished.
                await ingest_now(session, document.id)
                await session.refresh(document)
            print(f"  {name}: status={document.status} chunks={document.chunk_count}")

    await engine.dispose()

    print(
        "\nDone. Try it:\n"
        "  make run\n"
        f"  curl -s localhost:8000/api/v1/auth/login -H 'content-type: application/json' \\\n"
        f'    -d \'{{"email":"{DEMO_EMAIL}","password":"{DEMO_PASSWORD}"}}\'\n'
    )


if __name__ == "__main__":
    asyncio.run(main())
