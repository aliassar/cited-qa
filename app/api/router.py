"""
Aggregate router.

WHY THE /api/v1 PREFIX EXISTS FROM DAY ONE
-------------------------------------------
Versioning is cheap now and expensive later. Once real clients depend on
`/matters`, you cannot change its response shape without breaking them, and
retrofitting a version prefix means touching every consumer. With v1 in place,
a breaking change becomes `/api/v2/matters` running side by side while v1 is
deprecated on a schedule.
"""

from fastapi import APIRouter

from app.api.v1 import auth, documents, health, matters, questions

api_router = APIRouter()

# Health lives OUTSIDE the version prefix: load balancers and orchestrators
# should not have to care about your API version to know if you are alive.
api_router.include_router(health.router)

v1 = APIRouter(prefix="/api/v1")
v1.include_router(auth.router)
v1.include_router(matters.router)
v1.include_router(documents.router)
v1.include_router(questions.router)

api_router.include_router(v1)
