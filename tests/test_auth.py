"""
CareerPilot — Unit & Integration Tests for User Authentication & Registration (Phase 11)
Tests:
  - Password hashing and verification
  - User registration endpoint (valid, duplicate username, duplicate email, invalid format)
  - User login endpoint (valid credentials, wrong password, non-existent user)
  - User listing and database persistence
"""

import pytest
from fastapi.testclient import TestClient

from backend.auth import (
    create_session_token,
    hash_password,
    verify_password,
    verify_session_token,
)
from backend.main import app
from database.connection import get_db, Base, engine, SessionLocal
from database.models import User, Student
from database.repository import (
    create_user,
    get_user_by_email,
    get_user_by_identifier,
    get_user_by_username,
    list_users,
)


class TestPasswordSecurity:
    """Validate PBKDF2 password hashing & verification utilities."""

    def test_hash_and_verify_success(self):
        plain = "MySecretPass123!"
        hashed = hash_password(plain)
        assert hashed != plain
        assert "$" in hashed
        assert verify_password(plain, hashed) is True

    def test_verify_failure_wrong_password(self):
        plain = "CorrectPassword123"
        hashed = hash_password(plain)
        assert verify_password("WrongPassword456", hashed) is False

    def test_verify_empty_inputs(self):
        assert verify_password("", "some$hash") is False
        assert verify_password("pass", "") is False


class TestUserRepository:
    """Validate database operations for User model."""

    def test_create_and_find_user(self, db_session):
        username = "test_user_repo"
        email = "test_repo@example.com"
        
        # Clean up if exists from prior runs
        existing = get_user_by_username(db_session, username)
        if existing:
            db_session.delete(existing)
            db_session.commit()

        user = create_user(
            db=db_session,
            username=username,
            email=email,
            hashed_password=hash_password("Password123"),
            full_name="Test Candidate",
            target_technology="Python",
        )
        assert user.id is not None
        assert user.username == username
        assert user.email == email

        # Test lookup by username
        found_by_u = get_user_by_username(db_session, username)
        assert found_by_u is not None
        assert found_by_u.id == user.id

        # Test lookup by email
        found_by_e = get_user_by_email(db_session, email)
        assert found_by_e is not None
        assert found_by_e.id == user.id

        # Test lookup by identifier (username and email)
        assert get_user_by_identifier(db_session, username).id == user.id
        assert get_user_by_identifier(db_session, email).id == user.id


class TestAuthAPIEndpoints:
    """Validate FastAPI registration and login endpoints."""

    def test_register_new_user_success(self, client):
        unique_suffix = "auth_test_1"
        payload = {
            "username": f"candidate_{unique_suffix}",
            "email": f"{unique_suffix}@careerpilot.test",
            "password": "SecurePassword123",
            "full_name": "Priya Patel",
            "target_technology": "Java",
        }
        res = client.post("/api/auth/register", json=payload)
        assert res.status_code == 201
        data = res.json()
        assert data["success"] is True
        assert data["user"]["username"] == payload["username"]
        assert data["user"]["email"] == payload["email"]
        assert "token" in data and len(data["token"]) > 10

    def test_register_duplicate_username_fails(self, client):
        payload = {
            "username": "duplicate_candidate",
            "email": "dup1@careerpilot.test",
            "password": "SecurePassword123",
            "full_name": "First User",
            "target_technology": "Python",
        }
        res1 = client.post("/api/auth/register", json=payload)
        assert res1.status_code == 201

        # Second attempt with same username
        payload2 = {
            "username": "duplicate_candidate",
            "email": "different_email@careerpilot.test",
            "password": "SecurePassword123",
            "full_name": "Second User",
            "target_technology": "Python",
        }
        res2 = client.post("/api/auth/register", json=payload2)
        assert res2.status_code == 400
        assert "already taken" in res2.json()["detail"]

    def test_register_duplicate_email_fails(self, client):
        payload = {
            "username": "unique_u1",
            "email": "same_email@careerpilot.test",
            "password": "SecurePassword123",
            "full_name": "First User",
            "target_technology": "Python",
        }
        res1 = client.post("/api/auth/register", json=payload)
        assert res1.status_code == 201

        payload2 = {
            "username": "unique_u2",
            "email": "same_email@careerpilot.test",
            "password": "SecurePassword123",
            "full_name": "Second User",
            "target_technology": "Python",
        }
        res2 = client.post("/api/auth/register", json=payload2)
        assert res2.status_code == 400
        assert "already exists" in res2.json()["detail"]

    def test_login_success_with_username(self, client):
        username = "login_success_user"
        email = "login_success@careerpilot.test"
        pwd = "ValidPassword999"

        client.post("/api/auth/register", json={
            "username": username,
            "email": email,
            "password": pwd,
            "full_name": "Login Tester",
            "target_technology": "C++",
        })

        res = client.post("/api/auth/login", json={
            "username_or_email": username,
            "password": pwd,
        })
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert data["user"]["username"] == username
        assert "token" in data

    def test_login_success_with_email(self, client):
        username = "email_login_user"
        email = "email_login@careerpilot.test"
        pwd = "ValidPassword999"

        client.post("/api/auth/register", json={
            "username": username,
            "email": email,
            "password": pwd,
            "full_name": "Email Tester",
            "target_technology": "React / Node.js Full Stack",
        })

        res = client.post("/api/auth/login", json={
            "username_or_email": email,
            "password": pwd,
        })
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert data["user"]["username"] == username

    def test_login_wrong_password_fails(self, client):
        username = "wrong_pwd_user"
        client.post("/api/auth/register", json={
            "username": username,
            "email": "wrong_pwd@careerpilot.test",
            "password": "RightPassword123",
        })

        res = client.post("/api/auth/login", json={
            "username_or_email": username,
            "password": "WrongPassword999",
        })
        assert res.status_code == 401
        assert "Invalid username/email or password" in res.json()["detail"]

    def test_login_nonexistent_user_fails(self, client):
        res = client.post("/api/auth/login", json={
            "username_or_email": "does_not_exist_xyz",
            "password": "AnyPassword123",
        })
        assert res.status_code == 401
        assert "Invalid username/email or password" in res.json()["detail"]

    def test_list_users_endpoint(self, client):
        client.post("/api/auth/register", json={
            "username": "listed_user",
            "email": "listed@careerpilot.test",
            "password": "Password123",
            "full_name": "Listed User",
            "target_technology": "Python",
        })
        res = client.get("/api/auth/users")
        assert res.status_code == 200
        data = res.json()
        assert "total_users" in data
        assert isinstance(data["users"], list)
        assert data["total_users"] >= 1



class TestSessionSecurityAndLinkage:
    """Validate cryptographic session tokens, signature verification, and user-student linkage."""

    def test_session_token_creation_and_verification(self):
        token = create_session_token(user_id=42, username="alex_dev", expires_in_seconds=3600)
        assert token and "." in token

        valid, payload, error = verify_session_token(token)
        assert valid is True
        assert error is None
        assert payload["sub"] == 42
        assert payload["username"] == "alex_dev"
        assert "exp" in payload

    def test_session_token_tampering_fails(self):
        token = create_session_token(user_id=1, username="root")
        tampered = token[:-4] + "fake"

        valid, payload, error = verify_session_token(tampered)
        assert valid is False
        assert payload is None
        assert "Invalid token signature" in (error or "")

    def test_session_token_expiration_fails(self):
        # Generate token already expired by 10 seconds
        token = create_session_token(user_id=99, username="old_user", expires_in_seconds=-10)

        valid, payload, error = verify_session_token(token)
        assert valid is False
        assert payload is not None
        assert payload["username"] == "old_user"
        assert "expired" in (error or "").lower()

    def test_auth_verify_endpoint_success_and_invalid(self, client):
        reg_res = client.post("/api/auth/register", json={
            "username": "verified_user",
            "email": "verify@careerpilot.test",
            "password": "Password123",
            "full_name": "Verified User",
            "target_technology": "Python",
        })
        assert reg_res.status_code == 201
        token = reg_res.json()["token"]

        # 1. Valid token in header
        v_res = client.get("/api/auth/verify", headers={"Authorization": f"Bearer {token}"})
        assert v_res.status_code == 200
        v_data = v_res.json()
        assert v_data["valid"] is True
        assert v_data["user"]["username"] == "verified_user"

        # 2. Tampered token
        tampered_res = client.get("/api/auth/verify", headers={"Authorization": f"Bearer {token}bad"})
        assert tampered_res.status_code == 401

        # 3. Missing token
        no_auth_res = client.get("/api/auth/verify")
        assert no_auth_res.status_code == 401

    def test_student_user_id_bonding_on_registration(self, client, db_session):
        username = "bonded_candidate"
        email = "bonded@careerpilot.test"
        reg_res = client.post("/api/auth/register", json={
            "username": username,
            "email": email,
            "password": "Password123",
            "full_name": "Bonded Candidate",
            "target_technology": "Go",
        })
        assert reg_res.status_code == 201
        user_id = reg_res.json()["user"]["id"]

        # Verify Student record was automatically created and bound to user_id
        student = db_session.query(Student).filter(Student.user_id == user_id).first()
        assert student is not None
        assert student.user_id == user_id
        assert student.email == email
        assert student.target_technology == "Go"
