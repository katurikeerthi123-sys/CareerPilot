"""
CareerPilot — Authentication & Password Security Utilities

Provides cryptographically secure PBKDF2-HMAC-SHA256 password hashing,
constant-time verification, and session token generation and verification.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from typing import Any, Optional


def _b64encode(data: bytes) -> str:
    """Base64 URL-safe encoding without padding."""
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def _b64decode(data: str) -> bytes:
    """Base64 URL-safe decoding with auto padding restoration."""
    padding = 4 - (len(data) % 4)
    if padding and padding < 4:
        data += "=" * padding
    return base64.urlsafe_b64decode(data.encode("utf-8"))


def _sign(data: str, secret: str) -> str:
    """Compute HMAC-SHA256 signature encoded in URL-safe base64."""
    sig = hmac.new(secret.encode("utf-8"), data.encode("utf-8"), hashlib.sha256).digest()
    return _b64encode(sig)


def hash_password(password: str) -> str:
    """Hash password using PBKDF2-HMAC-SHA256 with a unique random salt."""
    salt = secrets.token_hex(16)
    key = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        iterations=100_000,
    )
    return f"{salt}${key.hex()}"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify plain password against PBKDF2 hash using constant-time comparison."""
    if not plain_password or not hashed_password:
        return False
    try:
        salt, key_hex = hashed_password.split("$", 1)
        test_key = hashlib.pbkdf2_hmac(
            "sha256",
            plain_password.encode("utf-8"),
            salt.encode("utf-8"),
            iterations=100_000,
        )
        return hmac.compare_digest(test_key.hex(), key_hex)
    except Exception:
        return False


def create_session_token(
    user_id: Optional[int] = None,
    username: Optional[str] = None,
    expires_in_seconds: Optional[int] = None,
    secret: Optional[str] = None,
) -> str:
    """
    Generate a cryptographically signed HMAC-SHA256 session token with expiration.

    Token format: <payload_base64url>.<signature_base64url>
    Payload: {sub: user_id, username: username, iat: now, exp: now + expires_in_seconds, nonce: hex}
    """
    if secret is None:
        try:
            from backend.config import get_settings
            secret = get_settings().auth_secret_key
        except Exception:
            secret = "careerpilot-session-signing-secret-key-2026"

    if expires_in_seconds is None:
        try:
            from backend.config import get_settings
            expires_in_seconds = get_settings().token_expiry_hours * 3600
        except Exception:
            expires_in_seconds = 7 * 24 * 3600  # 7 days

    now = int(time.time())
    payload = {
        "sub": user_id if user_id is not None else 0,
        "username": username or "anonymous",
        "iat": now,
        "exp": now + expires_in_seconds,
        "nonce": secrets.token_hex(8),
    }

    payload_json = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    payload_b64 = _b64encode(payload_json.encode("utf-8"))
    sig_b64 = _sign(payload_b64, secret)
    return f"{payload_b64}.{sig_b64}"


def verify_session_token(
    token: str,
    secret: Optional[str] = None,
) -> tuple[bool, Optional[dict[str, Any]], Optional[str]]:
    """
    Validate a cryptographic session token structure, signature, and expiration.

    Returns:
      (is_valid: bool, payload: Optional[dict], error_message: Optional[str])
    """
    if not token or not isinstance(token, str):
        return False, None, "Token missing or invalid type."

    parts = token.strip().split(".")
    if len(parts) != 2:
        return False, None, "Malformed token structure (expected payload.signature)."

    payload_b64, sig_b64 = parts

    if secret is None:
        try:
            from backend.config import get_settings
            secret = get_settings().auth_secret_key
        except Exception:
            secret = "careerpilot-session-signing-secret-key-2026"

    expected_sig = _sign(payload_b64, secret)
    if not hmac.compare_digest(sig_b64, expected_sig):
        return False, None, "Invalid token signature (token may be forged or tampered)."

    try:
        raw_bytes = _b64decode(payload_b64)
        payload = json.loads(raw_bytes.decode("utf-8"))
    except Exception as exc:
        return False, None, f"Failed to parse token payload: {exc}"

    exp = payload.get("exp")
    if exp is not None:
        now = int(time.time())
        if now > exp:
            return False, payload, "Token has expired."

    return True, payload, None
