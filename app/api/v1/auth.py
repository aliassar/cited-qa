"""
Auth endpoints.

Handlers are THIN on purpose: parse -> call service -> shape response. Business
rules live in `app/services/auth.py`. If a handler grows past ~20 lines, logic
has leaked into the transport layer.
"""

from __future__ import annotations

from fastapi import APIRouter, status

from app.api.deps import CurrentUser, RateLimited, SessionDep, SettingsDep
from app.schemas.auth import (
    LoginRequest,
    RefreshRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
)
from app.services import auth as auth_service

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,  # 201 + the created resource
    summary="Create an organisation and its first admin user",
    dependencies=[RateLimited],
)
async def register(
    payload: RegisterRequest, session: SessionDep, settings: SettingsDep
) -> UserResponse:
    user = await auth_service.register_organization(
        session,
        settings,
        org_name=payload.org_name,
        email=payload.email,
        full_name=payload.full_name,
        password=payload.password,
    )
    # Build the DTO explicitly. `response_model` would filter the ORM object
    # anyway, but being explicit makes the leak-proofing visible.
    return UserResponse.model_validate(user)


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Exchange credentials for an access + refresh token pair",
    dependencies=[RateLimited],  # the endpoint most in need of a limit
)
async def login(payload: LoginRequest, session: SessionDep, settings: SettingsDep) -> TokenResponse:
    user = await auth_service.authenticate(
        session, settings, email=payload.email, password=payload.password
    )
    access, refresh = await auth_service.issue_token_pair(session, settings, user)
    return TokenResponse(
        access_token=access,
        refresh_token=refresh,
        expires_in=settings.access_token_ttl_seconds,
    )


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Rotate a refresh token for a fresh pair",
    dependencies=[RateLimited],
)
async def refresh(
    payload: RefreshRequest, session: SessionDep, settings: SettingsDep
) -> TokenResponse:
    access, new_refresh = await auth_service.rotate_refresh_token(
        session, settings, payload.refresh_token
    )
    return TokenResponse(
        access_token=access,
        refresh_token=new_refresh,
        expires_in=settings.access_token_ttl_seconds,
    )


@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,  # 204: success, nothing to say
    summary="Revoke a refresh token",
)
async def logout(payload: RefreshRequest, session: SessionDep, settings: SettingsDep) -> None:
    await auth_service.revoke_refresh_token(session, settings, payload.refresh_token)


@router.get("/me", response_model=UserResponse, summary="The caller's own profile")
async def me(user: CurrentUser) -> UserResponse:
    return UserResponse.model_validate(user)
