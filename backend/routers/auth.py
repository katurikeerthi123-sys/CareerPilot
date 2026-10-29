"""
CareerPilot — User Authentication API Router

Exposes REST endpoints for:
  - POST /api/auth/register — Register new user account with hashed password
  - POST /api/auth/login    — Authenticate credentials and return session token
  - GET  /api/auth/verify   — Verify cryptographic session token & return profile
  - GET  /api/auth/users    — List registered users (profile summaries)
"""

from __future__ import annotations

import re
from typing import Any, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from loguru import logger
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.auth import (
    create_session_token,
    hash_password,
    verify_password,
    verify_session_token,
)
from database.connection import get_db
from database.models import User
from database.repository import (
    create_user,
    get_user_by_email,
    get_user_by_identifier,
    get_user_by_username,
    list_users,
    update_user_last_login,
)

router = APIRouter(prefix="/api/auth", tags=["Authentication & User Accounts"])

EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# ── Request / Response Schemas ────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50, description="Unique username")
    email: str = Field(..., min_length=5, max_length=150, description="Valid email address")
    password: str = Field(..., min_length=6, max_length=100, description="Password (at least 6 characters)")
    full_name: Optional[str] = Field(default=None, max_length=120, description="Student's full name")
    target_technology: Optional[str] = Field(default="Java", max_length=80, description="Preferred technology")


class LoginRequest(BaseModel):
    username_or_email: str = Field(..., min_length=2, max_length=150, description="Username or registered email")
    password: str = Field(..., min_length=1, max_length=100, description="Account password")


class AuthResponse(BaseModel):
    success: bool
    message: str
    token: Optional[str] = None
    user: Optional[dict[str, Any]] = None


class VerifyResponse(BaseModel):
    valid: bool
    message: str
    user: Optional[dict[str, Any]] = None
    expires_at: Optional[int] = None


# ── Dependency: Get Current Authenticated User ───────────────────────────────

def get_current_user(
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
) -> User:
    """Dependency for protected endpoints verifying Bearer token in header."""
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header required.",
        )
    parts = authorization.strip().split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization format. Expected 'Bearer <token>'.",
        )
    raw_token = parts[1]
    is_valid, payload, error_msg = verify_session_token(raw_token)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=error_msg or "Invalid session token.",
        )
    user_id = payload.get("sub") if payload else None
    user = db.query(User).filter(User.id == user_id).first() if user_id else None
    if not user and payload and "username" in payload:
        user = get_user_by_username(db, payload["username"])
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User associated with token not found.",
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is deactivated.",
        )
    return user


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/register",
    response_model=AuthResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user account",
)
def register_user(request: RegisterRequest, db: Session = Depends(get_db)):
    """Create a new user account with secure hashed password storage."""
    username_clean = request.username.strip().lower()
    email_clean = request.email.strip().lower()

    # 1. Validate username format
    if not re.match(r"^[a-zA-Z0-9_\-\.]+$", username_clean):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username may only contain letters, numbers, underscores, hyphens, and periods.",
        )

    # 2. Validate email format
    if not EMAIL_REGEX.match(email_clean):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please provide a valid email address.",
        )

    # 3. Check uniqueness
    if get_user_by_username(db, username_clean):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Username '{username_clean}' is already taken. Please choose another.",
        )

    if get_user_by_email(db, email_clean):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"An account with email '{email_clean}' already exists. Please log in.",
        )

    # 4. Hash password and persist user
    pwd_hash = hash_password(request.password)
    user = create_user(
        db=db,
        username=username_clean,
        email=email_clean,
        hashed_password=pwd_hash,
        full_name=request.full_name or request.username.strip(),
        target_technology=request.target_technology or "Java",
    )

    token = create_session_token(user_id=user.id, username=user.username)
    logger.info(f"User registration successful for '{user.username}'")

    return AuthResponse(
        success=True,
        message="Registration successful! Welcome to CareerPilot.",
        token=token,
        user=user.to_dict(),
    )


@router.post(
    "/login",
    response_model=AuthResponse,
    summary="Authenticate user and issue session token",
)
def login_user(request: LoginRequest, db: Session = Depends(get_db)):
    """Authenticate with username or email and password."""
    identifier = request.username_or_email.strip().lower()
    user = get_user_by_identifier(db, identifier)

    if not user:
        logger.warning(f"Failed login attempt: identifier '{identifier}' not found")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username/email or password.",
        )

    if not verify_password(request.password, user.hashed_password):
        logger.warning(f"Failed login attempt: incorrect password for '{user.username}'")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username/email or password.",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This account has been deactivated. Please contact support.",
        )

    update_user_last_login(db, user.id)
    token = create_session_token(user_id=user.id, username=user.username)
    logger.info(f"User login successful for '{user.username}'")

    return AuthResponse(
        success=True,
        message=f"Welcome back, {user.full_name or user.username}!",
        token=token,
        user=user.to_dict(),
    )


@router.get(
    "/verify",
    response_model=VerifyResponse,
    summary="Verify cryptographic session token and return user profile",
)
def verify_token(
    authorization: Optional[str] = Header(None),
    token: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    """
    Validates token from Authorization: Bearer <token> or query parameter ?token=<token>.
    Returns user details if active and valid.
    """
    raw_token = token
    if not raw_token and authorization:
        parts = authorization.strip().split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            raw_token = parts[1]
        elif len(parts) == 1:
            raw_token = parts[0]

    if not raw_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session token missing.",
        )

    is_valid, payload, error_msg = verify_session_token(raw_token)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=error_msg or "Invalid session token.",
        )

    user_id = payload.get("sub") if payload else None
    user = db.query(User).filter(User.id == user_id).first() if user_id else None
    if not user and payload and "username" in payload:
        user = get_user_by_username(db, payload["username"])

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User associated with token no longer exists.",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is deactivated.",
        )

    return VerifyResponse(
        valid=True,
        message="Token is valid and active.",
        user=user.to_dict(),
        expires_at=payload.get("exp") if payload else None,
    )


@router.get(
    "/users",
    summary="List all registered user summaries",
)
def get_registered_users(db: Session = Depends(get_db)):
    """Return list of all registered users without exposing password hashes."""
    users = list_users(db)
    return {
        "total_users": len(users),
        "users": [u.to_dict() for u in users],
    }
