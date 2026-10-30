"""
CareerPilot — Streamlit Frontend Dashboard (Phase 4 — RAG Integration)

A clean, responsive dashboard for students preparing for campus & tech placements.
Features:
  - Student setup (name, technology, days, placement goal)
  - RAG document management (upload, view status, re-index into ChromaDB)
  - Interactive semantic search & AI concept explanation grounded in study materials
  - AI-generated day-by-day roadmap with completion tracking
  - Progress metrics, recommended next task banner, and agent transcript
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

import requests
import streamlit as st

# ── Ensure project root is importable ─────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Sync Streamlit Secrets into os.environ (Cloud Deployment) ─────────────────
try:
    if hasattr(st, "secrets"):
        for key, val in st.secrets.items():
            if isinstance(val, dict):
                for sub_k, sub_v in val.items():
                    if sub_k not in os.environ:
                        os.environ[sub_k] = str(sub_v)
            elif key not in os.environ:
                os.environ[key] = str(val)
except Exception:
    pass

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CareerPilot — AI Placement Preparation",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Backend Configuration ─────────────────────────────────────────────────────
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")


@st.cache_resource(show_spinner=False)
def start_in_process_backend() -> bool:
    """Start FastAPI backend in a background daemon thread for standalone cloud deployments (e.g. Streamlit Cloud)."""
    # 1. Check if backend is already alive (e.g. standard local uvicorn or docker)
    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=1.5)
        if resp.status_code == 200:
            return True
    except Exception:
        pass

    # 2. Only auto-start if BACKEND_URL targets localhost or 127.0.0.1
    if "localhost" not in BACKEND_URL and "127.0.0.1" not in BACKEND_URL:
        return False

    try:
        import uvicorn
        from backend.main import app as fastapi_app

        server_config = uvicorn.Config(
            fastapi_app,
            host="127.0.0.1",
            port=8000,
            log_level="warning",
            access_log=False,
        )
        server = uvicorn.Server(server_config)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()

        # Poll until backend responds or timeout after 8 seconds
        for _ in range(16):
            time.sleep(0.5)
            try:
                resp = requests.get(f"{BACKEND_URL}/health", timeout=1)
                if resp.status_code == 200:
                    return True
            except Exception:
                pass
        return True
    except Exception as exc:
        print(f"[CareerPilot] Could not auto-start in-process backend: {exc}")
        return False


# Auto-start backend in standalone cloud container
start_in_process_backend()

# ── Session State Initialisation ──────────────────────────────────────────────
if "agent_data" not in st.session_state:
    st.session_state["agent_data"] = None

if "completed_days" not in st.session_state:
    st.session_state["completed_days"] = set()

if "search_results" not in st.session_state:
    st.session_state["search_results"] = None

if "explanation_result" not in st.session_state:
    st.session_state["explanation_result"] = None

if "last_evaluation" not in st.session_state:
    st.session_state["last_evaluation"] = None

if "student_answer_text" not in st.session_state:
    st.session_state["student_answer_text"] = ""

if "authenticated_user" not in st.session_state:
    st.session_state["authenticated_user"] = None

if "auth_token" not in st.session_state:
    st.session_state["auth_token"] = None


# ── API Helpers ───────────────────────────────────────────────────────────────

def check_backend_health() -> tuple[bool, Optional[dict[str, Any]]]:
    """Check if FastAPI backend is healthy and return status."""
    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=3)
        if resp.status_code == 200:
            return True, resp.json()
    except Exception:
        pass

    # Retry starting backend if running locally and currently offline
    if "localhost" in BACKEND_URL or "127.0.0.1" in BACKEND_URL:
        start_in_process_backend()
        try:
            resp = requests.get(f"{BACKEND_URL}/health", timeout=3)
            if resp.status_code == 200:
                return True, resp.json()
        except Exception:
            pass

    return False, None


def get_documents_status() -> dict[str, Any]:
    """Fetch current indexing status and document counts from ChromaDB."""
    try:
        resp = requests.get(f"{BACKEND_URL}/api/documents/status", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {"total_chunks_in_chroma": 0, "documents_on_disk": []}


def trigger_ingestion() -> tuple[bool, str]:
    """Trigger document loading and indexing into ChromaDB."""
    try:
        resp = requests.post(f"{BACKEND_URL}/api/documents/ingest", timeout=60)
        if resp.status_code == 200:
            data = resp.json()
            return True, data.get("message", "Ingestion completed.")
        return False, f"Server error: {resp.text}"
    except Exception as exc:
        return False, f"Ingestion request failed: {str(exc)}"


def call_agent_api(
    name: str,
    goal: str,
    technology: str,
    days: int,
    api_key: Optional[str] = None,
) -> tuple[bool, Optional[dict[str, Any]], Optional[str]]:
    """Invoke the CareerPilot backend agent endpoint."""
    effective_key = (
        api_key
        or st.session_state.get("gemini_api_key")
        or os.getenv("GEMINI_API_KEY", "").strip()
        or os.getenv("GOOGLE_API_KEY", "").strip()
    )
    payload = {
        "goal": goal.strip(),
        "student_name": name.strip() or "Student",
        "target_technology": technology,
        "available_days": days,
        "api_key": effective_key,
    }
    auth_user = st.session_state.get("authenticated_user")
    if isinstance(auth_user, dict) and auth_user.get("id"):
        payload["user_id"] = auth_user["id"]

    auth_token = st.session_state.get("auth_token")
    headers = {"Authorization": f"Bearer {auth_token}"} if auth_token else {}
    try:
        resp = requests.post(
            f"{BACKEND_URL}/api/agent",
            json=payload,
            headers=headers,
            timeout=120,
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get("success"):
                return True, data, None
            return False, None, data.get("error", "Agent returned an error.")
        return False, None, f"Server responded with HTTP {resp.status_code}: {resp.text}"
    except requests.exceptions.ConnectionError:
        return (
            False,
            None,
            "Cannot connect to CareerPilot backend. Please ensure the FastAPI server is running on "
            f"`{BACKEND_URL}`.",
        )
    except requests.exceptions.Timeout:
        return (
            False,
            None,
            "The preparation planning request timed out. Please try again with a concise goal.",
        )
    except Exception as exc:
        return False, None, f"Unexpected error: {str(exc)}"


def call_evaluate_api(
    answer: str,
    agent_data: dict[str, Any],
    api_key: Optional[str] = None,
) -> tuple[bool, Optional[dict[str, Any]], Optional[str]]:
    """Submit a student answer to the agentic evaluation and adaptation endpoint."""
    effective_key = (
        api_key
        or st.session_state.get("gemini_api_key")
        or os.getenv("GEMINI_API_KEY", "").strip()
        or os.getenv("GOOGLE_API_KEY", "").strip()
    )
    payload = {
        "student_answer": answer.strip(),
        "current_question": agent_data.get("current_question"),
        "question_details": agent_data.get("question_details") or {},
        "current_topic": agent_data.get("current_topic") or "General",
        "student_profile": agent_data.get("student_profile") or {},
        "weak_topics": agent_data.get("weak_topics") or [],
        "mastered_topics": agent_data.get("mastered_topics") or [],
        "topic_scores": agent_data.get("topic_scores") or {},
        "progress": agent_data.get("progress") or {},
        "session_id": agent_data.get("session_id"),
        "api_key": effective_key,
    }
    auth_user = st.session_state.get("authenticated_user")
    if isinstance(auth_user, dict) and auth_user.get("id"):
        payload["user_id"] = auth_user["id"]

    auth_token = st.session_state.get("auth_token")
    headers = {"Authorization": f"Bearer {auth_token}"} if auth_token else {}
    try:
        resp = requests.post(
            f"{BACKEND_URL}/api/agent/evaluate",
            json=payload,
            headers=headers,
            timeout=120,
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get("success"):
                return True, data, None
            return False, None, data.get("error", "Evaluation returned an error.")
        return False, None, f"Server responded with HTTP {resp.status_code}: {resp.text}"
    except requests.exceptions.ConnectionError:
        return False, None, f"Cannot connect to CareerPilot backend at `{BACKEND_URL}`."
    except requests.exceptions.Timeout:
        return False, None, "Evaluation request timed out. Please try again."
    except Exception as exc:
        return False, None, f"Evaluation request failed: {str(exc)}"


def get_student_history_api(name: str) -> list[dict[str, Any]]:
    """Fetch persistent answer and evaluation history from SQLite backend."""
    try:
        resp = requests.get(f"{BACKEND_URL}/api/students/{name.strip()}/history", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return []


def get_student_detail_api(name: str) -> Optional[dict[str, Any]]:
    """Fetch persistent student profile and progress record from SQLite."""
    try:
        resp = requests.get(f"{BACKEND_URL}/api/students/{name.strip()}", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def load_student_dashboard_state(student_name: str) -> Optional[dict[str, Any]]:
    """Automatically restore existing study plan, topics, and memory from SQLite backend."""
    clean_name = student_name.strip()
    if not clean_name:
        return None

    # 1. Try FastAPI REST endpoint
    try:
        resp = requests.get(f"{BACKEND_URL}/api/students/{clean_name}", timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            sess = data.get("latest_session")
            goal = data.get("active_goal") or {}
            prog = data.get("progress") or {}
            if sess and (sess.get("plan_structured") or sess.get("plan_markdown")):
                return {
                    "student_name": clean_name,
                    "student_profile": {
                        "name": clean_name,
                        "technology": goal.get("target_technology") or "Java",
                        "days_available": goal.get("days_available") or 7,
                        "role": "Candidate",
                        "difficulty_level": goal.get("target_difficulty") or "Intermediate",
                    },
                    "plan_structured": sess.get("plan_structured") or {},
                    "plan_markdown": sess.get("plan_markdown") or "",
                    "current_topic": sess.get("current_topic") or "General Preparation",
                    "recommended_next_task": sess.get("recommended_next_task") or "Review core concepts and practice questions.",
                    "next_action": sess.get("next_action") or "continue",
                    "weak_topics": prog.get("weak_topics") or [],
                    "mastered_topics": prog.get("mastered_topics") or [],
                    "completed_topics": prog.get("completed_topics") or [],
                    "progress": prog,
                }
    except Exception:
        pass

    # 2. Try direct SQLite fallback (guarantees restore on Streamlit Cloud)
    try:
        from database.connection import SessionLocal
        from database.models import Student, Goal, StudySession, Progress
        from sqlalchemy import func

        db = SessionLocal()
        try:
            st_rec = db.query(Student).filter(func.lower(Student.name) == clean_name.lower()).first()
            if not st_rec:
                return None

            sess = (
                db.query(StudySession)
                .filter(StudySession.student_id == st_rec.id)
                .order_by(StudySession.created_at.desc())
                .first()
            )
            if not sess or (not sess.plan_structured and not sess.plan_markdown):
                return None

            goal = (
                db.query(Goal)
                .filter(Goal.student_id == st_rec.id, Goal.is_active == True)
                .order_by(Goal.created_at.desc())
                .first()
            )
            prog = db.query(Progress).filter(Progress.student_id == st_rec.id).first()

            return {
                "student_name": st_rec.name,
                "student_profile": {
                    "name": st_rec.name,
                    "technology": (goal.target_technology if goal else st_rec.target_technology) or "Java",
                    "days_available": (goal.days_available if goal else st_rec.available_days) or 7,
                    "role": st_rec.current_role or "Candidate",
                    "difficulty_level": goal.target_difficulty if goal else "Intermediate",
                },
                "plan_structured": sess.plan_structured or {},
                "plan_markdown": sess.plan_markdown or "",
                "current_topic": sess.current_topic or "General Preparation",
                "recommended_next_task": sess.recommended_next_task or "Review core concepts and practice questions.",
                "next_action": sess.next_action or "continue",
                "weak_topics": prog.weak_topics if prog else [],
                "mastered_topics": prog.mastered_topics if prog else [],
                "completed_topics": prog.completed_topics if prog else [],
                "progress": prog.to_dict() if prog else {},
            }
        finally:
            db.close()
    except Exception:
        return None


def call_placement_question_api(
    topic: str,
    technology: str = "Java",
    question_type: str = "conceptual",
    difficulty: str = "Medium",
) -> tuple[bool, Optional[dict[str, Any]], Optional[str]]:
    """Fetch a placement question formatted as MCQ, Coding, or Conceptual."""
    try:
        resp = requests.post(
            f"{BACKEND_URL}/api/placement/question",
            json={
                "topic": topic.strip(),
                "technology": technology,
                "question_type": question_type,
                "difficulty": difficulty,
            },
            timeout=60,
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get("success"):
                return True, data, None
            return False, None, data.get("error", "Failed to generate question.")
        return False, None, f"HTTP {resp.status_code}: {resp.text}"
    except Exception as exc:
        return False, None, f"Request failed: {str(exc)}"


def call_mcq_check_api(
    selected_option: str,
    correct_option: str,
    explanation: str = "",
    question_id: Optional[int] = None,
) -> tuple[bool, Optional[dict[str, Any]], Optional[str]]:
    """Instant grading and explanation for MCQ practice questions."""
    try:
        resp = requests.post(
            f"{BACKEND_URL}/api/placement/mcq/check",
            json={
                "question_id": question_id,
                "selected_option": selected_option,
                "correct_option": correct_option,
                "explanation": explanation,
            },
            timeout=10,
        )
        if resp.status_code == 200:
            return True, resp.json(), None
        return False, None, f"HTTP {resp.status_code}: {resp.text}"
    except Exception as exc:
        return False, None, f"MCQ check failed: {str(exc)}"


def call_readiness_report_api(
    student_name: str,
    target_technology: str = "Java",
    target_role: str = "Software Engineer",
) -> tuple[bool, Optional[dict[str, Any]], Optional[str]]:
    """Generate final comprehensive placement readiness report."""
    try:
        resp = requests.post(
            f"{BACKEND_URL}/api/placement/readiness-report",
            json={
                "student_name": student_name.strip(),
                "target_technology": target_technology,
                "target_role": target_role,
            },
            timeout=60,
        )
        if resp.status_code == 200:
            return True, resp.json(), None
        return False, None, f"HTTP {resp.status_code}: {resp.text}"
    except Exception as exc:
        return False, None, f"Report generation failed: {str(exc)}"


def call_interview_start_api(
    student_name: str,
    target_role: str = "Software Development Engineer (SDE-1)",
    technology: str = "Java",
    total_questions: int = 5,
    difficulty: str = "Medium",
) -> tuple[bool, Optional[dict[str, Any]], Optional[str]]:
    """Initialize mock interview session and pose Turn 1 question."""
    try:
        resp = requests.post(
            f"{BACKEND_URL}/api/interview/start",
            json={
                "student_name": student_name.strip(),
                "target_role": target_role,
                "technology": technology,
                "total_questions": total_questions,
                "difficulty": difficulty,
            },
            timeout=60,
        )
        if resp.status_code == 201:
            return True, resp.json(), None
        return False, None, f"HTTP {resp.status_code}: {resp.text}"
    except Exception as exc:
        return False, None, f"Interview launch failed: {str(exc)}"


def call_interview_answer_api(
    session_id: str,
    question_index: int,
    student_answer: str,
) -> tuple[bool, Optional[dict[str, Any]], Optional[str]]:
    """Submit answer to interviewer, obtain tech/comm scores and next question or final summary."""
    try:
        resp = requests.post(
            f"{BACKEND_URL}/api/interview/answer",
            json={
                "session_id": session_id,
                "question_index": question_index,
                "student_answer": student_answer.strip(),
            },
            timeout=90,
        )
        if resp.status_code == 200:
            return True, resp.json(), None
        return False, None, f"HTTP {resp.status_code}: {resp.text}"
    except Exception as exc:
        return False, None, f"Answer evaluation failed: {str(exc)}"


def call_interview_report_api(session_id: str) -> tuple[bool, Optional[dict[str, Any]], Optional[str]]:
    """Retrieve finalized mock interview report."""
    try:
        resp = requests.get(f"{BACKEND_URL}/api/interview/{session_id}/report", timeout=15)
        if resp.status_code == 200:
            return True, resp.json(), None
        return False, None, f"HTTP {resp.status_code}: {resp.text}"
    except Exception as exc:
        return False, None, f"Report fetch failed: {str(exc)}"







# ── Authentication API Helpers ────────────────────────────────────────────────

def _get_auth_modules():
    """Ensure database.models, database.repository, and backend.auth are fresh in sys.modules."""
    import importlib
    import sys

    import database.connection as db_conn
    import database.models as db_models
    import database.repository as db_repo
    import backend.auth as b_auth

    # Hot-reload stale modules if they were cached before auth commits
    if not hasattr(db_models, "User"):
        db_models = importlib.reload(db_models)
    if not hasattr(db_repo, "create_user"):
        db_repo = importlib.reload(db_repo)
    if not hasattr(b_auth, "hash_password"):
        b_auth = importlib.reload(b_auth)

    # Ensure SQLite tables exist
    try:
        db_models.Base.metadata.create_all(bind=db_conn.engine)
    except Exception:
        pass

    return db_conn, db_models, db_repo, b_auth


def call_register_api(
    username: str,
    email: str,
    password: str,
    full_name: Optional[str] = None,
    target_technology: str = "Java",
) -> tuple[bool, Optional[dict[str, Any]], Optional[str]]:
    """Register a new candidate account with the backend (with in-process fallback for cloud)."""
    payload = {
        "username": username.strip(),
        "email": email.strip(),
        "password": password,
        "full_name": full_name.strip() if full_name else None,
        "target_technology": target_technology,
    }
    
    # 1. Attempt HTTP call to FastAPI backend
    try:
        resp = requests.post(f"{BACKEND_URL}/api/auth/register", json=payload, timeout=15)
        if resp.status_code == 201:
            data = resp.json() if resp.content else {}
            if data.get("success"):
                return True, data, None
        elif resp.status_code != 404:
            # Real validation error from backend (e.g., duplicate username, bad email)
            data = resp.json() if resp.content else {}
            detail = data.get("detail") or data.get("message") or f"HTTP {resp.status_code}"
            return False, None, str(detail)
    except Exception:
        pass

    # 2. Fallback to direct in-process SQLite persistence (guarantees success on Streamlit Cloud)
    try:
        import re
        db_conn, db_models, db_repo, b_auth = _get_auth_modules()

        clean_u = username.strip().lower()
        clean_e = email.strip().lower()
        clean_name = full_name.strip() if full_name else clean_u

        if not re.match(r"^[a-zA-Z0-9_.-]+$", clean_u):
            return False, None, "Username may only contain letters, numbers, underscores, hyphens, and periods."
        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", clean_e):
            return False, None, "Please provide a valid email address."

        db = db_conn.SessionLocal()
        try:
            if hasattr(db_repo, "get_user_by_username") and db_repo.get_user_by_username(db, clean_u):
                return False, None, f"Username '{clean_u}' is already taken. Please choose another."
            if hasattr(db_repo, "get_user_by_email") and db_repo.get_user_by_email(db, clean_e):
                return False, None, f"An account with email '{clean_e}' already exists. Please log in."

            pwd_hash = b_auth.hash_password(password)
            user = db_repo.create_user(
                db=db,
                username=clean_u,
                email=clean_e,
                hashed_password=pwd_hash,
                full_name=clean_name,
                target_technology=target_technology or "Java",
            )
            token = b_auth.create_session_token()
            return True, {
                "success": True,
                "message": "Registration successful! Welcome to CareerPilot.",
                "token": token,
                "user": user.to_dict(),
            }, None
        finally:
            db.close()
    except Exception as exc:
        return False, None, f"Registration failed: {str(exc)}"


def call_login_api(
    username_or_email: str,
    password: str,
) -> tuple[bool, Optional[dict[str, Any]], Optional[str]]:
    """Authenticate existing candidate credentials with backend (with in-process fallback for cloud)."""
    payload = {
        "username_or_email": username_or_email.strip(),
        "password": password,
    }
    
    # 1. Attempt HTTP call to FastAPI backend
    try:
        resp = requests.post(f"{BACKEND_URL}/api/auth/login", json=payload, timeout=15)
        if resp.status_code == 200:
            data = resp.json() if resp.content else {}
            if data.get("success"):
                return True, data, None
        elif resp.status_code not in (404, 502, 503):
            data = resp.json() if resp.content else {}
            detail = data.get("detail") or data.get("message") or f"HTTP {resp.status_code}"
            return False, None, str(detail)
    except Exception:
        pass

    # 2. Fallback to direct in-process SQLite verification
    try:
        db_conn, db_models, db_repo, b_auth = _get_auth_modules()

        clean_id = username_or_email.strip().lower()
        db = db_conn.SessionLocal()
        try:
            user = db_repo.get_user_by_identifier(db, clean_id)
            if not user or not b_auth.verify_password(password, user.hashed_password):
                return False, None, "Invalid username/email or password."
            if not user.is_active:
                return False, None, "This account has been deactivated. Please contact support."

            if hasattr(db_repo, "update_user_last_login"):
                db_repo.update_user_last_login(db, user.id)
            token = b_auth.create_session_token()
            return True, {
                "success": True,
                "message": f"Welcome back, {user.full_name or user.username}!",
                "token": token,
                "user": user.to_dict(),
            }, None
        finally:
            db.close()
    except Exception as exc:
        return False, None, f"Login failed: {str(exc)}"


def render_auth_page():
    """Render Login & Registration portal for candidate access."""
    st.markdown(
        """
        <div style="text-align: center; padding: 1.5rem 0 1.2rem 0;">
            <h1 style="font-size: 2.8rem; margin-bottom: 0.2rem;">🚀 CareerPilot</h1>
            <p style="font-size: 1.15rem; color: #555; max-width: 680px; margin: 0 auto; line-height: 1.6;">
                <strong>Agentic AI for Personalized Placement Preparation</strong><br>
                Please sign in or register to access your personalized preparation roadmaps, adaptive practice,
                ChromaDB study materials, and AI mock interviews.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        st.info("🎯 **Custom Roadmaps**\n\nDay-by-day plan tailored to your timeline.")
    with f2:
        st.info("📚 **RAG Knowledge Base**\n\nCurated placement technical sheets.")
    with f3:
        st.info("🤖 **Adaptive Tutor**\n\nEvaluates answers and remediates weak topics.")
    with f4:
        st.info("🎙️ **Mock Interviews**\n\nRealistic technical hiring simulations.")

    st.markdown("---")

    col_l, col_center, col_r = st.columns([1, 2.2, 1])
    with col_center:
        auth_tab_login, auth_tab_register = st.tabs(["🔐 Sign In", "📝 Create Account"])

        with auth_tab_login:
            st.markdown("### 🔐 Sign In to Your Account")
            st.caption("Enter your registered credentials to access your dashboard.")
            with st.form("login_form", clear_on_submit=False):
                login_id = st.text_input("Username or Email", placeholder="e.g. arjun or arjun@example.com")
                login_pwd = st.text_input("Password", type="password", placeholder="Enter your password")
                login_submit = st.form_submit_button("🚀 Sign In to CareerPilot", type="primary", use_container_width=True)

                if login_submit:
                    if not login_id.strip() or not login_pwd:
                        st.error("Please enter both username/email and password.")
                    else:
                        with st.spinner("Authenticating credentials with backend..."):
                            ok, resp_data, err_msg = call_login_api(login_id, login_pwd)
                            if ok and resp_data:
                                st.session_state["authenticated_user"] = resp_data.get("user")
                                st.session_state["auth_token"] = resp_data.get("token")
                                # Clear previous in-memory state so user gets a clean session
                                st.session_state["agent_data"] = None
                                st.session_state["completed_days"] = set()
                                st.session_state["last_evaluation"] = None
                                st.session_state["student_answer_text"] = ""
                                st.session_state["active_interview_session"] = None
                                st.session_state["interview_final_report"] = None
                                st.toast(f"Welcome back, {resp_data['user'].get('full_name')}!", icon="👋")
                                time.sleep(0.5)
                                st.rerun()
                            else:
                                st.error(f"❌ {err_msg}")

            st.markdown("<div style='text-align: center; margin: 0.8rem 0 0.4rem 0; color: #888;'>── OR ──</div>", unsafe_allow_html=True)
            if st.button("⚡ One-Click Demo Access (Quick Test)", use_container_width=True):
                ok, resp_data, _ = call_login_api("demo_candidate", "CareerPilot123!")
                if not ok:
                    ok, resp_data, _ = call_register_api(
                        username="demo_candidate",
                        email="demo@careerpilot.ai",
                        password="CareerPilot123!",
                        full_name="Demo Candidate",
                        target_technology="Java",
                    )
                if ok and resp_data:
                    st.session_state["authenticated_user"] = resp_data.get("user")
                    st.session_state["auth_token"] = resp_data.get("token")
                    st.session_state["agent_data"] = None
                    st.session_state["completed_days"] = set()
                    st.session_state["last_evaluation"] = None
                    st.session_state["student_answer_text"] = ""
                    st.session_state["active_interview_session"] = None
                    st.session_state["interview_final_report"] = None
                    st.toast("Logged in as Demo Candidate!", icon="⚡")
                    time.sleep(0.5)
                    st.rerun()

        with auth_tab_register:
            st.markdown("### 📝 Register New Candidate Account")
            st.caption("Your details and placement progress will be safely stored in the SQLite backend.")
            with st.form("register_form", clear_on_submit=False):
                reg_name = st.text_input("Full Name", placeholder="e.g. Arjun Sharma")
                reg_user = st.text_input("Username", placeholder="e.g. arjun_sharma (alphanumeric)")
                reg_email = st.text_input("Email Address", placeholder="e.g. arjun@university.edu")
                reg_tech = st.selectbox(
                    "Primary Placement Track / Technology",
                    [
                        "Java",
                        "Python",
                        "C++",
                        "JavaScript / TypeScript",
                        "React / Node.js Full Stack",
                        "Data Structures & Algorithms (DSA)",
                        "Data Science & Machine Learning",
                        "SQL & Database Engineering",
                        "Go / Golang",
                        "Other / Custom",
                    ],
                )
                reg_p1 = st.text_input("Create Password", type="password", placeholder="Minimum 6 characters")
                reg_p2 = st.text_input("Confirm Password", type="password", placeholder="Re-enter password")
                reg_submit = st.form_submit_button("✨ Register & Start Preparation", type="primary", use_container_width=True)

                if reg_submit:
                    if not reg_user.strip() or not reg_email.strip() or not reg_p1:
                        st.error("Please fill in all required fields (Username, Email, Password).")
                    elif len(reg_p1) < 6:
                        st.error("Password must be at least 6 characters long.")
                    elif reg_p1 != reg_p2:
                        st.error("Passwords do not match. Please verify and try again.")
                    else:
                        with st.spinner("Registering your candidate account in backend..."):
                            ok, resp_data, err_msg = call_register_api(
                                username=reg_user,
                                email=reg_email,
                                password=reg_p1,
                                full_name=reg_name or reg_user,
                                target_technology=reg_tech,
                            )
                            if ok and resp_data:
                                st.session_state["authenticated_user"] = resp_data.get("user")
                                st.session_state["auth_token"] = resp_data.get("token")
                                # Clear previous in-memory state so user gets a clean session
                                st.session_state["agent_data"] = None
                                st.session_state["completed_days"] = set()
                                st.session_state["last_evaluation"] = None
                                st.session_state["student_answer_text"] = ""
                                st.session_state["active_interview_session"] = None
                                st.session_state["interview_final_report"] = None
                                st.success("🎉 Account created successfully! Launching dashboard...")
                                time.sleep(0.6)
                                st.rerun()
                            else:
                                st.error(f"❌ {err_msg}")


# ── Authentication Gate ───────────────────────────────────────────────────────
if not st.session_state.get("authenticated_user"):
    render_auth_page()
    st.stop()

# ── Auto-Restore Candidate's Saved Preparation Track ──────────────────────────
if not st.session_state.get("agent_data"):
    _curr_u = st.session_state.get("authenticated_user") or {}
    _user_id = str(_curr_u.get("id")) if _curr_u.get("id") else None
    _cand_name = _curr_u.get("full_name") or _curr_u.get("username")
    _saved_track = None
    if _user_id:
        _saved_track = load_student_dashboard_state(_user_id)
    if not _saved_track and _cand_name:
        _saved_track = load_student_dashboard_state(_cand_name)
    if _saved_track:
        st.session_state["agent_data"] = _saved_track

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://raw.githubusercontent.com/feathericons/feather/master/icons/compass.svg",
        width=48,
    )
    st.title("CareerPilot")
    st.caption("Agentic AI for Placement Prep • Phase 4 (RAG)")
    st.markdown("---")

    # Current Authenticated Candidate
    curr_user = st.session_state.get("authenticated_user") or {}
    st.markdown(f"**👤 {curr_user.get('full_name') or curr_user.get('username')}**")
    st.caption(f"@{curr_user.get('username')} • {curr_user.get('email')}")
    st.markdown(f"**Track:** `{curr_user.get('target_technology', 'Java')}`")
    if st.button("🚪 Log Out", use_container_width=True, key="btn_logout_sidebar"):
        st.session_state["authenticated_user"] = None
        st.session_state["auth_token"] = None
        st.session_state["agent_data"] = None
        st.session_state["active_interview_session"] = None
        st.session_state["interview_final_report"] = None
        st.rerun()
    st.markdown("---")

    # Backend Status
    backend_ok, health_info = check_backend_health()
    if backend_ok:
        st.success("🟢 Backend Connected", icon="✅")
    else:
        st.error(
            "🔴 Backend Offline\n\nStart via:\n`uvicorn backend.main:app --reload`",
            icon="⚠️",
        )

    # API Key Configuration: hidden from visitors if pre-configured on server / secrets
    def _has_secret(k: str) -> bool:
        try:
            return hasattr(st, "secrets") and k in st.secrets
        except Exception:
            return False

    server_has_key = bool(
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or _has_secret("GEMINI_API_KEY")
        or _has_secret("GOOGLE_API_KEY")
    )

    if not server_has_key:
        active_key = st.session_state.get("gemini_api_key", "")
        with st.expander("🔑 Gemini API Key (Admin Setup)", expanded=not bool(active_key)):
            user_key = st.text_input(
                "API Key",
                value=active_key,
                type="password",
                placeholder="AQ... or AIzaSy...",
                help="Set GEMINI_API_KEY in Streamlit Cloud Secrets so all visitors can use the app without entering any key.",
            )
            if user_key and user_key.strip():
                cleaned_key = user_key.strip()
                os.environ["GEMINI_API_KEY"] = cleaned_key
                os.environ["GOOGLE_API_KEY"] = cleaned_key
                st.session_state["gemini_api_key"] = cleaned_key

    # ── Preparation Setup Section ─────────────────────────────────────────────
    st.markdown("### 🎯 Preparation Setup")

    # 1. Student Name
    default_student_name = curr_user.get("full_name") or curr_user.get("username") or "Arjun Sharma"
    student_name_input = st.text_input(
        "Your Name",
        value=default_student_name,
        placeholder="e.g. Alex, Priya, Arjun",
        help="Used to personalise your study plan and interview feedback.",
    )

    # 2. Target Technology
    tech_options = [
        "Java",
        "Python",
        "C++",
        "JavaScript / TypeScript",
        "React / Node.js Full Stack",
        "Data Structures & Algorithms (DSA)",
        "Data Science & Machine Learning",
        "SQL & Database Engineering",
        "Go / Golang",
        "Other / Custom",
    ]
    user_tech = curr_user.get("target_technology", "Java")
    default_tech_idx = tech_options.index(user_tech) if user_tech in tech_options else 0
    selected_tech = st.selectbox(
        "Target Technology",
        options=tech_options,
        index=default_tech_idx,
        help="The primary language or domain for your placement interview.",
    )

    if selected_tech == "Other / Custom":
        custom_tech = st.text_input("Specify Technology", placeholder="e.g. Rust, Kotlin, AWS")
        effective_tech = custom_tech.strip() or "General Software Engineering"
    else:
        effective_tech = selected_tech

    target_tech = effective_tech

    # 3. Available Preparation Days
    prep_days = st.slider(
        "Available Preparation Days",
        min_value=1,
        max_value=30,
        value=7,
        help="Select how many days you have until your placement interview.",
    )

    # 4. Placement Goal
    default_goal = f"I have a {effective_tech} placement interview in {prep_days} days. Prepare me with core concepts, coding patterns, and mock questions."
    goal_input = st.text_area(
        "Placement Goal Description",
        value=default_goal,
        height=100,
        help="Be as specific as you like (e.g. target company, weak areas, expected role).",
    )

    # 5. Action Buttons
    col_btn1, col_btn2 = st.columns([2, 1])
    with col_btn1:
        start_btn = st.button("🚀 Start AI Prep", type="primary", use_container_width=True)
    with col_btn2:
        reset_btn = st.button("🔄 Reset", use_container_width=True)

    if reset_btn:
        st.session_state["agent_data"] = None
        st.session_state["completed_days"] = set()
        st.session_state["search_results"] = None
        st.session_state["explanation_result"] = None
        st.session_state["last_evaluation"] = None
        st.session_state["student_answer_text"] = ""
        st.rerun()

    # ── RAG Study Materials Section in Sidebar ─────────────────────────────────
    st.markdown("---")
    st.markdown("### 📚 Study Materials (RAG)")
    doc_stats = get_documents_status()
    total_chunks = doc_stats.get("total_chunks_in_chroma", 0)
    files_on_disk = doc_stats.get("documents_on_disk", [])

    st.caption(f"**ChromaDB:** {total_chunks} chunks indexed | **Files:** {len(files_on_disk)}")
    if files_on_disk:
        with st.expander("📄 Files on Disk"):
            for f in files_on_disk:
                st.markdown(f"- `{f}`")

    # Document Upload
    uploaded_file = st.file_uploader(
        "Upload Study Guide / Notes",
        type=["txt", "md", "pdf", "docx"],
        help="Upload company interview sheets or subject notes to index into ChromaDB.",
    )
    if uploaded_file is not None:
        upload_btn = st.button("💾 Save Uploaded Document", use_container_width=True)
        if upload_btn:
            doc_dir = PROJECT_ROOT / "data" / "documents"
            doc_dir.mkdir(parents=True, exist_ok=True)
            save_path = doc_dir / uploaded_file.name
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Saved `{uploaded_file.name}`. Click 'Index Documents' below.")

    ingest_btn = st.button("📥 Index Documents into ChromaDB", use_container_width=True)
    if ingest_btn:
        with st.spinner("Chunking & embedding documents into ChromaDB..."):
            ok, msg = trigger_ingestion()
            if ok:
                st.success(msg)
                time.sleep(1)
                st.rerun()
            else:
                st.error(msg)


# ── Handle Start Submission ───────────────────────────────────────────────────
if start_btn:
    if not goal_input.strip():
        st.error("Please enter a placement preparation goal.")
    else:
        with st.spinner("🤖 CareerPilot Agent is retrieving your study materials and building your plan..."):
            success, data, error_msg = call_agent_api(
                name=student_name_input,
                goal=goal_input,
                technology=effective_tech,
                days=prep_days,
            )
            if success and data:
                st.session_state["agent_data"] = data
                st.session_state["completed_days"] = set()
                st.toast("🎉 Personalized preparation plan ready!", icon="🚀")
            else:
                if "API key required" in str(error_msg):
                    st.error(
                        "🔑 **Gemini API Key Required**\n\n"
                        "Please paste your Gemini API key in the left sidebar under **'🔑 Gemini API Key'**, "
                        "or add `GEMINI_API_KEY = \"your-key\"` in your Streamlit Cloud **App Settings ➔ Secrets**."
                    )
                else:
                    st.error(f"❌ Failed to generate plan: {error_msg}")


# ── Main Content Area ─────────────────────────────────────────────────────────
agent_data = st.session_state.get("agent_data")

if not agent_data:
    # ── Welcome / Hero State ──────────────────────────────────────────────────
    st.title("🚀 CareerPilot: Agentic Placement Preparation")
    _curr_cand = st.session_state.get("authenticated_user") or {}
    _c_name = _curr_cand.get("full_name") or _curr_cand.get("username")
    _past_items = get_student_history_api(_c_name) if _c_name else []
    if _past_items:
        st.success(
            f"Welcome back, **{_c_name}**! You have **{len(_past_items)} recorded evaluation(s)** in your persistent SQLite memory.",
            icon="📊",
        )
        with st.expander(f"📜 View Your Past Evaluation Records ({len(_past_items)} Attempted)", expanded=False):
            for _idx, _item in enumerate(_past_items, 1):
                _isp = _item.get("is_passed", False)
                _sc = _item.get("score", 0.0)
                _b = "✅ Pass" if _isp else "⚠️ Review Needed"
                st.markdown(f"**{_idx}. [{_b}] {_item.get('topic', 'General')}** — Score: `{_sc}/10`")
                st.markdown(f"> *Q: {_item.get('question_text')}*")
                if _item.get("feedback"):
                    st.caption(f"Feedback: {_item.get('feedback')}")
        st.markdown("---")
    st.markdown(
        """
        Welcome! CareerPilot is an **agentic AI placement tutor with Retrieval-Augmented Generation (RAG)**.
        It pairs your study documents with LLM reasoning to generate an adaptive curriculum, retrieve
        precise concepts, and guide your daily practice.
        """
    )

    st.markdown("### 💡 Core Agentic Workflow")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            """
            #### 1. Goal & Profile
            Extracts target technology, role, and available preparation window.
            """
        )
    with col2:
        st.markdown(
            """
            #### 2. RAG Retrieval
            Semantic search pulls key concepts from your study materials in ChromaDB.
            """
        )
    with col3:
        st.markdown(
            """
            #### 3. Plan & Alignment
            Constructs a tailored daily schedule aligned with your actual notes.
            """
        )
    with col4:
        st.markdown(
            """
            #### 4. Practice & Adapt
            Evaluates your answers, detects weak areas, and updates the roadmap.
            """
        )

    st.info(
        "👈 **Get started:** Enter your goals and click **'🚀 Start AI Prep'** in the sidebar, or explore the sample study materials.",
        icon="ℹ️",
    )

    # Study Materials Preview on Welcome Screen
    st.markdown("### 📚 Study Materials Ready in ChromaDB")
    doc_stats_welcome = get_documents_status()
    st.markdown(
        f"- **Indexed Chunks:** `{doc_stats_welcome.get('total_chunks_in_chroma', 0)}` chunks\n"
        f"- **Available Documents:** {', '.join([f'`{d}`' for d in doc_stats_welcome.get('documents_on_disk', [])]) or 'None'}"
    )

else:
    # ── Active Dashboard State ────────────────────────────────────────────────
    student_profile = agent_data.get("student_profile") or {}
    plan_structured = agent_data.get("plan_structured") or {}
    plan_markdown = agent_data.get("plan_markdown") or ""
    current_topic = agent_data.get("current_topic") or "General Preparation"
    recommended_next_task = (
        agent_data.get("recommended_next_task")
        or "Review core fundamentals and practice introductory problems."
    )
    student_name = agent_data.get("student_name") or "Student"
    technology = student_profile.get("technology", "Technology")
    target_tech = technology
    role = student_profile.get("role", "Software Engineer")
    days_available = student_profile.get("days_available", 7)
    difficulty_level = student_profile.get("difficulty_level", "Intermediate")
    daily_schedule = plan_structured.get("daily_schedule", [])

    # Header Title
    st.title(f"🎓 Preparation Dashboard: {student_name}")
    st.caption(f"Role: **{role}** • Target: **{technology}** • Difficulty: **{difficulty_level}** • Powered by **ChromaDB RAG**")

    # ── Key Metrics Overview ──────────────────────────────────────────────────
    weak_topics = agent_data.get("weak_topics") or []
    mastered_topics = agent_data.get("mastered_topics") or []
    next_action_val = agent_data.get("next_action") or "continue"
    progress_val = agent_data.get("progress") or {}
    total_days = len(daily_schedule) or days_available
    completed_count = len(st.session_state["completed_days"])
    overall_pct = progress_val.get("overall_progress_percent", int((completed_count / total_days) * 100) if total_days > 0 else 0)

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric(label="🎯 Current Topic", value=current_topic[:18] + ("…" if len(current_topic) > 18 else ""))
    with m2:
        st.metric(label="🏆 Mastered", value=f"{len(mastered_topics)} Topics")
    with m3:
        st.metric(
            label="⚠️ Weak Areas",
            value=f"{len(weak_topics)} Topics",
            delta="Needs Review" if weak_topics else "None",
            delta_color="inverse" if weak_topics else "normal",
        )
    with m4:
        st.metric(label="📈 Mastery Progress", value=f"{overall_pct}%")
    with m5:
        action_label = {
            "repeat_weak_topic": "Remediate Weakness",
            "next_topic": "Next Topic",
            "finish": "Prep Complete",
            "continue": "Practice",
        }.get(next_action_val, next_action_val)
        st.metric(label="⚡ Next Action", value=action_label)

    # ── Actionable Recommended Next Task Banner ───────────────────────────────
    st.markdown("---")
    st.info(
        f"**👉 Recommended Next Task:**  \n"
        f"**{recommended_next_task}**  \n\n"
        f"*Focus Topic:* `{current_topic}`",
        icon="💡",
    )

    # ── Progress Bar ──────────────────────────────────────────────────────────
    st.progress(overall_pct / 100.0, text=f"Mastery Progress: {overall_pct}% ({len(mastered_topics)} mastered, {len(weak_topics)} weak)")

    # ── Dashboard Tabs ────────────────────────────────────────────────────────
    tab_plan, tab_practice, tab_interview, tab_report, tab_rag, tab_history, tab_progress, tab_agent_log = st.tabs([
        "📋 Preparation Plan",
        "🎯 Placement Practice (MCQ / Coding / Conceptual)",
        "🎙️ Mock Interview Agent",
        "🏆 Placement Readiness Report",
        "📚 Study Materials (RAG)",
        "📜 Persistent History (SQLite)",
        "📊 Progress & Checklist",
        "🤖 Agent Decisions & State",
    ])



    with tab_plan:
        st.subheader("🗓️ Day-by-Day Preparation Plan")
        if daily_schedule:
            for item in daily_schedule:
                day_num = item.get("day", 1)
                is_done = day_num in st.session_state["completed_days"]
                status_icon = "✅" if is_done else "📌"

                with st.expander(
                    f"{status_icon} Day {day_num}: {item.get('focus', 'Daily Study')} (~{item.get('estimated_hours', 2)} hrs)",
                    expanded=(day_num == 1),
                ):
                    col_d1, col_d2 = st.columns([3, 1])
                    with col_d1:
                        if item.get("goals"):
                            st.markdown("**🎯 Goals for Today:**")
                            for g in item["goals"]:
                                st.markdown(f"- {g}")

                        if item.get("topics"):
                            st.markdown("**📚 Sub-Topics to Master:**")
                            for t in item["topics"]:
                                st.markdown(f"- `{t}`")

                        if item.get("practice"):
                            st.markdown(f"**💻 Practice Exercise:** {item['practice']}")

                    with col_d2:
                        st.markdown("**Status**")
                        toggle_done = st.checkbox(
                            "Mark Day Complete",
                            value=is_done,
                            key=f"check_day_{day_num}",
                        )
                        if toggle_done and not is_done:
                            st.session_state["completed_days"].add(day_num)
                            st.rerun()
                        elif not toggle_done and is_done:
                            st.session_state["completed_days"].remove(day_num)
                            st.rerun()
        else:
            st.markdown(plan_markdown)

        # Resources & Tips
        st.markdown("---")
        col_res, col_tip = st.columns(2)
        with col_res:
            st.markdown("### 📚 Key Resources")
            resources = plan_structured.get("key_resources", [])
            if resources:
                for r in resources:
                    st.markdown(f"- {r}")
            else:
                st.markdown("- Recommended online documentations and practice platforms.")

        with col_tip:
            st.markdown("### 💡 Placement Success Tips")
            tips = plan_structured.get("success_tips", [])
            if tips:
                for tip in tips:
                    st.markdown(f"- {tip}")
            else:
                st.markdown("- Practice whiteboard coding and articulate your thought process.")

    with tab_practice:
        st.subheader("🎯 Placement Practice & Adaptive Learning Loop")
        st.caption(
            "Solve targeted placement questions across multiple formats (MCQs, coding challenges, conceptual deep-dives). "
            "CareerPilot evaluates your performance, logs diagnostics to persistent memory, and adaptively prescribes remediation."
        )

        # ── Mode 1: Multi-Format Placement Question Generator ────────────────
        with st.expander("⚡ Launch On-Demand Practice Question (MCQ / Coding / Conceptual)", expanded=False):
            st.markdown("Generate placement-grade interview questions on any chosen topic and format:")
            f_col1, f_col2, f_col3 = st.columns([2, 2, 1])
            with f_col1:
                gen_format = st.selectbox(
                    "Question Format:",
                    ["🔘 Multiple Choice (MCQ)", "💻 Coding Challenge", "🎯 Conceptual Deep Dive"],
                    key="gen_question_format",
                )
            with f_col2:
                gen_topic = st.text_input(
                    "Target Topic:",
                    value=current_topic or "Java Collections",
                    key="gen_question_topic",
                )
            with f_col3:
                gen_diff = st.selectbox(
                    "Difficulty:",
                    ["Medium", "Easy", "Hard"],
                    key="gen_question_diff",
                )

            if st.button("🎲 Generate Multi-Format Question", type="secondary", use_container_width=True):
                format_key = "mcq" if "MCQ" in gen_format else ("coding" if "Coding" in gen_format else "conceptual")
                with st.spinner(f"Generating {gen_diff} {format_key.upper()} placement question for '{gen_topic}'..."):
                    ok_q, q_resp, err_q = call_placement_question_api(
                        topic=gen_topic,
                        technology=target_tech,
                        question_type=format_key,
                        difficulty=gen_diff,
                    )
                    if ok_q and q_resp:
                        st.session_state["active_placement_question"] = q_resp
                        st.session_state["mcq_result"] = None
                        st.toast(f"Generated {gen_diff} {format_key.upper()} question!", icon="✨")
                        st.rerun()
                    else:
                        st.error(f"Failed to generate question: {err_q}")

        # Active Multi-Format Placement Question Display
        active_pq = st.session_state.get("active_placement_question")
        if active_pq:
            pq_type = active_pq.get("question_type", "conceptual")
            pq_topic = active_pq.get("topic", "General")
            pq_diff = active_pq.get("difficulty", "Medium")
            st.markdown("---")
            st.markdown(f"### 📝 Active Practice: `{pq_topic}` ({pq_diff} | {pq_type.upper()})")

            # ── MCQ Render ──────────────────────────────────────────────────
            if pq_type == "mcq" and active_pq.get("mcq_details"):
                mcq = active_pq["mcq_details"]
                st.info(f"**Question:**\n\n{active_pq.get('question_text', '')}")

                opts = mcq.get("options", [])
                radio_options = [f"{opt.get('label')}: {opt.get('text')}" for opt in opts]

                selected_opt = st.radio(
                    "Select the best answer:",
                    options=radio_options,
                    key="mcq_selected_radio",
                )

                if st.button("✅ Check MCQ Answer", type="primary"):
                    chosen_label = selected_opt.split(":")[0].strip() if selected_opt else "A"
                    correct_label = mcq.get("correct_option", "A")
                    with st.spinner("Checking answer..."):
                        ok_chk, chk_res, err_chk = call_mcq_check_api(
                            question_id=active_pq.get("question_id"),
                            selected_option=chosen_label,
                            correct_option=correct_label,
                            explanation=mcq.get("explanation", ""),
                        )
                        if ok_chk and chk_res:
                            st.session_state["mcq_result"] = chk_res
                            st.rerun()
                        else:
                            st.error(f"Check failed: {err_chk}")

                mcq_res = st.session_state.get("mcq_result")
                if mcq_res:
                    if mcq_res.get("is_correct"):
                        st.success(f"🎉 **Correct!** (Selected: `{mcq_res.get('selected_option')}`)", icon="✅")
                    else:
                        st.error(f"❌ **Incorrect.** You selected `{mcq_res.get('selected_option')}`, but the correct answer is `{mcq_res.get('correct_option')}`.", icon="⚠️")
                    st.info(f"**Explanation:** {mcq_res.get('explanation')}")

            # ── Coding Challenge Render ──────────────────────────────────────
            elif pq_type == "coding" and active_pq.get("coding_details"):
                cd = active_pq["coding_details"]
                st.markdown(f"#### 💻 Problem Statement\n{cd.get('problem_statement')}")

                if cd.get("constraints"):
                    with st.expander("⚙️ Constraints"):
                        for c in cd["constraints"]:
                            st.markdown(f"- `{c}`")

                # Test Cases
                tc_list = cd.get("sample_test_cases", [])
                if tc_list:
                    st.markdown("##### 🧪 Sample Test Cases")
                    tc_cols = st.columns(min(len(tc_list), 3))
                    for i, tc in enumerate(tc_list[:3]):
                        with tc_cols[i]:
                            st.markdown(f"**Case {i+1}**")
                            st.code(f"Input:  {tc.get('input')}\nOutput: {tc.get('expected_output')}", language="text")

                st.markdown("##### ✍️ Starter Code & Implementation")
                starter = cd.get("starter_code") or "// Write your solution here\n"
                code_solution = st.text_area(
                    "Your Code Implementation:",
                    value=starter,
                    height=200,
                    key="coding_submission_text",
                )

                if st.button("🚀 Submit Code Solution to Evaluator", type="primary"):
                    if not code_solution.strip() or code_solution.strip() == starter.strip():
                        st.warning("Please implement your solution before submitting.")
                    else:
                        with st.spinner("CareerPilot is analyzing code correctness, edge cases, and time complexity..."):
                            ok_eval, eval_resp, err_eval = call_evaluate_api(code_solution, agent_data)
                            if ok_eval and eval_resp:
                                st.session_state["agent_data"] = eval_resp
                                st.session_state["last_evaluation"] = eval_resp.get("evaluation")
                                st.toast("Code evaluated successfully!", icon="💻")
                                st.rerun()
                            else:
                                st.error(f"Evaluation error: {err_eval}")

            # ── Conceptual Practice Question Render ──────────────────────────
            else:
                st.info(f"**Question:**\n\n{active_pq.get('question_text', '')}")
                if active_pq.get("hints"):
                    with st.expander("💡 Hints"):
                        for h in active_pq["hints"]:
                            st.markdown(f"- {h}")

                conc_answer = st.text_area(
                    "Your Technical Explanation:",
                    height=140,
                    placeholder="Explain mechanisms, internal working, and runtime trade-offs...",
                    key="custom_conceptual_answer",
                )
                if st.button("🚀 Evaluate Technical Explanation", type="primary"):
                    if not conc_answer.strip():
                        st.warning("Please type your answer.")
                    else:
                        with st.spinner("Evaluating explanation..."):
                            ok_eval, eval_resp, err_eval = call_evaluate_api(conc_answer, agent_data)
                            if ok_eval and eval_resp:
                                st.session_state["agent_data"] = eval_resp
                                st.session_state["last_evaluation"] = eval_resp.get("evaluation")
                                st.toast("Answer evaluated successfully!", icon="🎯")
                                st.rerun()
                            else:
                                st.error(f"Evaluation error: {err_eval}")

        # ── Mode 2: Scheduled Agentic Curriculum Question ─────────────────────
        st.markdown("---")
        st.markdown("### 🤖 Agentic Learning Loop Practice")

        current_question = agent_data.get("current_question")
        q_details = agent_data.get("question_details") or {}
        q_difficulty = q_details.get("difficulty", "Medium")
        q_category = q_details.get("category", "Conceptual / Coding")
        q_hints = q_details.get("hints", [])
        q_remedial = q_details.get("is_remedial", False)

        if current_question:
            badge_icon = "🔴" if q_remedial else "🔵"
            practice_type = "Remedial Targeted Practice" if q_remedial else "Curriculum Question"
            st.markdown(f"#### {badge_icon} {practice_type} • Topic: `{current_topic}`")

            # Formatted Question Box
            st.info(f"**Interview Question ({q_difficulty} | {q_category}):**\n\n{current_question}")

            if q_hints:
                with st.expander("💡 Hints & Guidance"):
                    for h in q_hints:
                        st.markdown(f"- {h}")

            # Answer Input Area
            student_answer = st.text_area(
                "Your Technical Answer / Explanation / Code:",
                value=st.session_state.get("student_answer_text", ""),
                height=150,
                placeholder="Type your explanation or code solution here. Be specific about mechanisms, time complexity, and edge cases...",
                key="student_practice_answer",
            )

            col_submit, col_clear = st.columns([2, 1])
            with col_submit:
                submit_ans = st.button("🚀 Submit Answer to AI Evaluator", type="primary", use_container_width=True)
            with col_clear:
                clear_ans = st.button("Clear Answer", use_container_width=True)
                if clear_ans:
                    st.session_state["student_answer_text"] = ""
                    st.rerun()

            if submit_ans:
                if not student_answer.strip():
                    st.warning("Please type your answer before submitting.")
                else:
                    with st.spinner("🤖 CareerPilot Evaluator is grading your answer, detecting weaknesses, and updating your roadmap..."):
                        ok, eval_resp, err = call_evaluate_api(student_answer, agent_data)
                        if ok and eval_resp:
                            st.session_state["agent_data"] = eval_resp
                            st.session_state["last_evaluation"] = eval_resp.get("evaluation")
                            st.session_state["student_answer_text"] = ""
                            st.toast("Evaluation complete! The agent has adapted your preparation loop.", icon="🎯")
                            st.rerun()
                        else:
                            st.error(f"Evaluation failed: {err}")
        else:
            st.info("No practice question is currently active. The curriculum may be complete or generating.")

        # ── Display Latest Diagnostic Evaluation ─────────────────────────────
        last_eval = st.session_state.get("last_evaluation") or agent_data.get("evaluation")
        if last_eval:
            st.markdown("---")
            st.markdown("### 📋 Latest AI Evaluation & Diagnostic Report")

            score = float(last_eval.get("score", 0.0))
            is_passed = last_eval.get("is_passed", False) or score >= 7.0
            verdict = last_eval.get("verdict", "Evaluated")
            feedback = last_eval.get("feedback", "")
            strengths = last_eval.get("strengths", [])
            areas = last_eval.get("areas_for_improvement", [])
            ideal = last_eval.get("ideal_answer", "")
            remediation = agent_data.get("remediation_notes")
            next_action = agent_data.get("next_action")

            c_score, c_verdict, c_action = st.columns(3)
            with c_score:
                st.metric(
                    "Score",
                    f"{score:.1f} / 10",
                    delta="Passed (>= 7.0)" if is_passed else "Below Threshold (< 7.0)",
                    delta_color="normal" if is_passed else "inverse",
                )
            with c_verdict:
                st.metric("Verdict", verdict)
            with c_action:
                action_display = {
                    "repeat_weak_topic": "⚠️ Remediation Required",
                    "next_topic": "✅ Advanced to Next Topic",
                    "finish": "🏆 All Topics Mastered",
                }.get(next_action, next_action or "Review")
                st.metric("Agent Adaptive Decision", action_display)

            if feedback:
                st.info(f"**Diagnostic Feedback:** {feedback}")

            col_eval1, col_eval2 = st.columns(2)
            with col_eval1:
                st.markdown("#### ✅ Demonstrated Strengths")
                if strengths:
                    for s in strengths:
                        st.markdown(f"- {s}")
                else:
                    st.caption("No specific strengths noted.")

            with col_eval2:
                st.markdown("#### ⚠️ Areas for Improvement")
                if areas:
                    for a in areas:
                        st.markdown(f"- {a}")
                else:
                    st.caption("No major areas for improvement noted.")

            if ideal:
                with st.expander("📖 View Model / Reference Answer"):
                    st.markdown(ideal)

            if remediation:
                st.warning(f"**Adaptive Remediation Guidance:** {remediation}", icon="💡")

    with tab_interview:
        st.subheader("🎙️ AI Mock Interviewer & Hiring Committee Simulator")
        st.caption(
            "Experience an authentic technical placement interview. The Interviewer Agent asks progressive questions, "
            "adapts based on your previous answers, and the Evaluator Agent scores both technical depth and communication structure in real-time."
        )

        # ── State 1: Configuration & Launch ──────────────────────────────────
        if "active_interview_session" not in st.session_state:
            st.session_state["active_interview_session"] = None
        if "interview_final_report" not in st.session_state:
            st.session_state["interview_final_report"] = None
        if "last_interview_eval" not in st.session_state:
            st.session_state["last_interview_eval"] = None

        active_int_session = st.session_state.get("active_interview_session")
        int_report = st.session_state.get("interview_final_report")

        if not active_int_session and not int_report:
            st.markdown("#### ⚙️ Configure Your Mock Interview Session")
            int_c1, int_c2, int_c3 = st.columns([2, 1, 1])
            with int_c1:
                interview_role = st.selectbox(
                    "Target Role:",
                    [
                        "Software Development Engineer (SDE-1)",
                        "Backend Engineer",
                        "Full Stack Developer",
                        "Java Platform Specialist",
                        "Associate Software Engineer",
                    ],
                    key="int_cfg_role",
                )
            with int_c2:
                interview_q_count = st.slider("Number of Questions:", min_value=3, max_value=6, value=4, key="int_cfg_count")
            with int_c3:
                interview_diff = st.selectbox("Interview Rigor:", ["Medium", "Hard", "Progressive"], key="int_cfg_diff")

            st.info(
                "💡 **How the Mock Interview Agent Works:**\n"
                "1. **Interviewer Agent** formulates real-world questions from fundamentals to system edge cases.\n"
                "2. **Evaluator Agent** scores your technical accuracy (0-10) and communication clarity (0-10).\n"
                "3. **Hiring Committee Finalizer** calculates your composite Readiness Score and delivers a hiring decision."
            )

            if st.button("🎙️ Launch Live Mock Interview", type="primary", use_container_width=True):
                with st.spinner("Staff Interviewer Agent is preparing your customized placement interview..."):
                    ok_i, int_start_resp, err_i = call_interview_start_api(
                        student_name=student_name,
                        target_role=interview_role,
                        technology=target_tech,
                        total_questions=interview_q_count,
                        difficulty=interview_diff,
                    )
                    if ok_i and int_start_resp:
                        st.session_state["active_interview_session"] = int_start_resp
                        st.session_state["interview_final_report"] = None
                        st.session_state["last_interview_eval"] = None
                        st.toast("Mock Interview Session Started!", icon="🎙️")
                        st.rerun()
                    else:
                        st.error(f"Failed to launch interview: {err_i}")

        # ── State 2: Active Question Turn ────────────────────────────────────
        elif active_int_session and not int_report:
            curr_q_idx = active_int_session.get("question_index", 1)
            total_int_q = active_int_session.get("total_questions", 4)
            q_topic = active_int_session.get("topic", "Technical Deep Dive")
            q_diff = active_int_session.get("difficulty", "Medium")
            preface = active_int_session.get("interviewer_preface")

            st.progress(curr_q_idx / float(total_int_q), text=f"Question {curr_q_idx} of {total_int_q} ({q_topic})")

            # Interviewer Question Card
            st.markdown(f"### 👔 Staff Interviewer • Question {curr_q_idx} of {total_int_q}")
            if preface:
                st.caption(f"🗣️ *\"{preface}\"*")

            st.info(f"**Question ({q_diff} | `{q_topic}`):**\n\n{active_int_session.get('question_text')}")

            if active_int_session.get("hints"):
                with st.expander("💡 Interviewer Hint"):
                    for h in active_int_session["hints"]:
                        st.markdown(f"- {h}")

            # Candidate Answer Area
            candidate_ans = st.text_area(
                "Your Technical Answer & Articulation:",
                height=180,
                placeholder="Structure your response: 1) Core definition/mechanism, 2) Internal working & memory/concurrency behavior, 3) Time/space trade-offs, 4) Real-world example...",
                key="mock_interview_candidate_answer",
            )

            col_sub_int, col_quit_int = st.columns([3, 1])
            with col_sub_int:
                submit_int_btn = st.button("🎤 Submit Answer to Interviewer", type="primary", use_container_width=True)
            with col_quit_int:
                quit_int_btn = st.button("Cancel Session", use_container_width=True)
                if quit_int_btn:
                    st.session_state["active_interview_session"] = None
                    st.session_state["interview_final_report"] = None
                    st.session_state["last_interview_eval"] = None
                    st.rerun()

            if submit_int_btn:
                if not candidate_ans.strip():
                    st.warning("Please type your response before submitting to the interviewer.")
                else:
                    with st.spinner("🤖 Evaluator Agent is scoring technical depth and communication clarity..."):
                        ok_ans, ans_resp, err_ans = call_interview_answer_api(
                            session_id=active_int_session.get("session_id"),
                            question_index=curr_q_idx,
                            student_answer=candidate_ans,
                        )
                        if ok_ans and ans_resp:
                            eval_res = ans_resp.get("evaluation")
                            st.session_state["last_interview_eval"] = eval_res

                            if ans_resp.get("is_finished"):
                                st.session_state["active_interview_session"] = None
                                st.session_state["interview_final_report"] = ans_resp.get("final_summary")
                                st.toast("Interview completed! Generating hiring committee scorecard...", icon="🎉")
                                st.rerun()
                            else:
                                st.session_state["active_interview_session"] = eval_res.get("next_question")
                                st.toast(f"Turn {curr_q_idx} scored! Moving to Question {curr_q_idx + 1}...", icon="✅")
                                st.rerun()
                        else:
                            st.error(f"Failed to submit answer: {err_ans}")

            # Previous Turn Scorecard
            last_turn_eval = st.session_state.get("last_interview_eval")
            if last_turn_eval:
                st.markdown("---")
                st.markdown("#### 📝 Previous Turn Evaluation")
                e1, e2, e3 = st.columns(3)
                with e1:
                    st.metric("Technical Score", f"{last_turn_eval.get('technical_score', 0):.1f} / 10")
                with e2:
                    st.metric("Communication Score", f"{last_turn_eval.get('communication_score', 0):.1f} / 10")
                with e3:
                    st.metric("Turn Verdict", last_turn_eval.get("verdict", "Evaluated"))

                st.markdown(f"**Feedback:** {last_turn_eval.get('feedback')}")
                if last_turn_eval.get("strengths"):
                    st.success(f"**Strengths:** {', '.join(last_turn_eval['strengths'])}")
                if last_turn_eval.get("areas_for_improvement"):
                    st.warning(f"**Areas for Improvement:** {', '.join(last_turn_eval['areas_for_improvement'])}")

        # ── State 3: Final Hiring Committee Scorecard ────────────────────────
        if int_report:
            st.markdown("---")
            st.success("🎉 **Mock Interview Session Completed!**", icon="🏆")
            st.markdown("### 📋 Lead Hiring Committee Assessment & Placement Readiness Scorecard")

            f_score = int_report.get("overall_interview_score", 0)
            f_verdict = int_report.get("readiness_verdict", "Hire")
            f_tech = int_report.get("technical_score", 0)
            f_comm = int_report.get("communication_score", 0)

            # Executive Scorecard Metrics
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.metric("Hiring Decision", f_verdict)
            with k2:
                st.metric("Overall Score", f"{f_score:.1f}%")
            with k3:
                st.metric("Technical Score", f"{f_tech:.1f}%")
            with k4:
                st.metric("Communication Score", f"{f_comm:.1f}%")

            # Topic-wise performance
            topic_perf_list = int_report.get("topic_wise_performance", [])
            if topic_perf_list:
                st.markdown("#### 📊 Topic-Wise Evaluation Breakdown")
                int_table = []
                for tp in topic_perf_list:
                    int_table.append({
                        "Topic": tp.get("topic"),
                        "Technical (10)": f"{tp.get('average_technical_score', 0):.1f}",
                        "Communication (10)": f"{tp.get('average_communication_score', 0):.1f}",
                        "Verdict": tp.get("verdict"),
                        "Observations": tp.get("observations"),
                    })
                st.table(int_table)

            # Strengths & Weaknesses
            s_col1, s_col2 = st.columns(2)
            with s_col1:
                st.markdown("#### 🌟 Demonstrated Strengths")
                for s in int_report.get("strong_areas", []):
                    st.markdown(f"- ✅ {s}")

            with s_col2:
                st.markdown("#### ⚠️ Identified Growth Areas")
                for w in int_report.get("weak_areas", []):
                    st.markdown(f"- 🔴 {w}")

            # Recommendations
            st.markdown("#### 🎯 Placement Interview Recommendations")
            for r in int_report.get("recommendations", []):
                st.markdown(f"- 💡 {r}")

            # Executive Debrief Notes
            st.markdown("#### 👨‍💼 Senior Interviewer Panel Debrief")
            st.info(int_report.get("interviewer_notes", "Good candidate performance overall."))

            # Restart Button
            if st.button("🔄 Start New Mock Interview Session", type="secondary"):
                st.session_state["active_interview_session"] = None
                st.session_state["interview_final_report"] = None
                st.session_state["last_interview_eval"] = None
                st.rerun()

    with tab_report:
        st.subheader("🏆 Comprehensive Placement Readiness Report")
        st.caption(
            "Synthesize your full preparation history, topic mastery index, and diagnostic evaluations "
            "into an executive-level placement assessment with target company tier matching and a prioritized action plan."
        )

        rep_col1, rep_col2, rep_col3 = st.columns([2, 2, 1])
        with rep_col1:
            report_role = st.selectbox(
                "Target Job Role:",
                [
                    "Software Development Engineer (SDE-1)",
                    "Full Stack Developer",
                    "Backend Engineer",
                    "Java Platform Specialist",
                    "Graduate Trainee / Associate Software Engineer",
                ],
                key="readiness_target_role",
            )
        with rep_col2:
            st.text_input("Candidate Name:", value=student_name, disabled=True)
        with rep_col3:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            gen_rep_btn = st.button("📊 Generate Report", type="primary", use_container_width=True)

        if gen_rep_btn:
            with st.spinner(f"Compiling complete SQLite history and generating Placement Readiness Assessment for {student_name}..."):
                ok_r, rep_data, err_r = call_readiness_report_api(
                    student_name=student_name,
                    target_technology=target_tech,
                    target_role=report_role,
                )
                if ok_r and rep_data:
                    st.session_state["placement_readiness_report"] = rep_data
                    st.toast("Placement Readiness Report generated successfully!", icon="🏆")
                    st.rerun()
                else:
                    st.error(f"Failed to generate report: {err_r}")

        # Display Report
        active_report = st.session_state.get("placement_readiness_report")
        if active_report:
            st.markdown("---")
            score_val = active_report.get("overall_readiness_score", 0)
            level_val = active_report.get("readiness_level", "Developing")

            # Executive Scorecard
            r1, r2, r3, r4 = st.columns(4)
            with r1:
                st.metric("Readiness Index", f"{score_val}%", delta=f"{level_val}")
            with r2:
                st.metric("Mastered Topics", active_report.get("total_topics_mastered", 0))
            with r3:
                st.metric("Weak Areas Flagged", active_report.get("total_weak_topics", 0))
            with r4:
                st.metric("Evaluations Logged", active_report.get("total_questions_attempted", 0))

            # Company Fit Tags
            companies = active_report.get("company_fit", [])
            if companies:
                st.markdown("#### 🏢 Suitable Company Profiles & Hiring Tiers")
                tag_cols = st.columns(min(len(companies), 4))
                for idx, c_tag in enumerate(companies[:4]):
                    with tag_cols[idx]:
                        st.success(f"🎯 **{c_tag}**")

            # Placement Advisor Verdict
            st.markdown("#### 👨‍💼 Placement Advisor Executive Summary")
            verdict_text = active_report.get("placement_advisor_verdict", "")
            st.info(f"💡 *\"{verdict_text}\"*")

            # Topic Competency Matrix Table
            competencies = active_report.get("competencies", [])
            if competencies:
                st.markdown("#### 📊 Curriculum Competency Breakdown")
                table_rows = []
                for comp in competencies:
                    table_rows.append({
                        "Topic": comp.get("topic"),
                        "Status": "🟢 " + comp.get("status") if comp.get("status") == "Mastered" else ("🔴 " + comp.get("status") if comp.get("status") == "Weak" else "🟡 " + comp.get("status")),
                        "Attempts": comp.get("attempts_count", 0),
                        "Avg Score": f"{comp.get('average_score', 0.0):.1f} / 10",
                        "Interview Advice": comp.get("recommendation", ""),
                    })
                st.table(table_rows)

            # Strengths & Growth Areas
            c_str, c_gro = st.columns(2)
            with c_str:
                st.markdown("#### 🌟 Key Demonstrated Strengths")
                for s in active_report.get("demonstrated_strengths", []):
                    st.markdown(f"- ✅ {s}")

            with c_gro:
                st.markdown("#### 🚀 Critical Growth Areas")
                for g in active_report.get("critical_growth_areas", []):
                    st.markdown(f"- ⚠️ {g}")

            # Prioritized Action Plan
            action_steps = active_report.get("action_plan_next_steps", [])
            if action_steps:
                st.markdown("#### 📋 Prioritized 3-Step Action Plan")
                for idx, step in enumerate(action_steps, 1):
                    st.markdown(f"**Step {idx}:** {step}")

            st.caption(f"Report generated at: `{active_report.get('generated_at')}`")
        else:
            st.info(
                "Click **'Generate Report'** above to compile your complete preparation metrics into "
                "an official Placement Readiness Report with hiring tier recommendations.",
                icon="ℹ️",
            )

    with tab_rag:
        st.subheader("🔍 Semantic Study Material Search & RAG Tutor")
        st.markdown(
            "Query your indexed documents directly or generate an in-depth interview explanation "
            "grounded in your materials."
        )

        query_col1, query_col2 = st.columns([3, 1])
        with query_col1:
            rag_query = st.text_input(
                "Enter concept or interview question:",
                value="HashMap collision resolution and TREEIFY_THRESHOLD",
                placeholder="e.g. JVM garbage collection, Polymorphism vs Overriding",
            )
        with query_col2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            search_click = st.button("🔎 Search Notes", use_container_width=True)

        explain_click = st.button("🧠 Explain Grounded in Documents", type="secondary")

        if search_click and rag_query.strip():
            with st.spinner("Searching ChromaDB vector store..."):
                try:
                    resp = requests.post(
                        f"{BACKEND_URL}/api/documents/query",
                        json={"query": rag_query.strip(), "top_k": 3},
                        timeout=15,
                    )
                    if resp.status_code == 200:
                        st.session_state["search_results"] = resp.json()
                        st.session_state["explanation_result"] = None
                    else:
                        st.error(f"Search failed: {resp.text}")
                except Exception as exc:
                    st.error(f"Error querying backend: {exc}")

        if explain_click and rag_query.strip():
            with st.spinner("Generating document-grounded explanation with Gemini..."):
                try:
                    resp = requests.post(
                        f"{BACKEND_URL}/api/documents/explain",
                        json={"query": rag_query.strip()},
                        timeout=30,
                    )
                    if resp.status_code == 200:
                        st.session_state["explanation_result"] = resp.json()
                        st.session_state["search_results"] = None
                    else:
                        st.error(f"Explanation failed: {resp.text}")
                except Exception as exc:
                    st.error(f"Error generating explanation: {exc}")

        # Display Search Results
        search_res = st.session_state.get("search_results")
        if search_res:
            st.markdown(f"#### 📄 Retrieved Chunks ({search_res.get('total_retrieved', 0)} matches)")
            for idx, chunk in enumerate(search_res.get("chunks", []), 1):
                with st.expander(f"Chunk {idx} — Source: `{chunk.get('source')}`", expanded=(idx == 1)):
                    st.markdown(chunk.get("content", ""))

        # Display Explanation Results
        explain_res = st.session_state.get("explanation_result")
        if explain_res:
            grounded = explain_res.get("grounded_in_documents")
            if grounded:
                st.success("✅ Grounded in your uploaded study materials", icon="📚")
            else:
                st.info("ℹ️ Answered using general industry knowledge", icon="💡")

            st.markdown("### 📝 Interview Explanation")
            st.markdown(explain_res.get("explanation", ""))

            with st.expander("🔍 View Context Chunks Used"):
                st.text(explain_res.get("context_used", "None"))

    with tab_history:
        st.subheader("📜 Persistent Memory & Historical Evaluations")
        st.caption(
            "CareerPilot stores all your attempted questions, answers, and evaluations in a local "
            "SQLite database (`database/careerpilot.db`). The agent consults this memory across sessions "
            "to recall your past weak topics and mastery trajectory."
        )

        history_items = get_student_history_api(student_name)
        student_details = get_student_detail_api(student_name) or {}
        prog_detail = student_details.get("progress") or {}

        # Top summary metrics
        h1, h2, h3, h4 = st.columns(4)
        with h1:
            st.metric("Total Questions Attempted", len(history_items))
        with h2:
            avg_score = prog_detail.get("average_score") or (
                round(sum(h["score"] for h in history_items) / len(history_items), 2)
                if history_items else 0.0
            )
            st.metric("Average Score", f"{avg_score:.1f} / 10")
        with h3:
            st.metric("Mastered Topics", len(mastered_topics))
        with h4:
            st.metric("Persistent Storage", "SQLite Active", delta="careerpilot.db")

        st.markdown("---")
        if history_items:
            st.markdown(f"#### 📝 Chronological Evaluation History ({len(history_items)} Records)")
            for idx, item in enumerate(history_items, 1):
                is_p = item.get("is_passed", False)
                score_val = item.get("score", 0.0)
                badge = "🟢 Pass" if is_p else "🔴 Remediation Flagged"
                timestamp = item.get("submitted_at", "")[:19].replace("T", " ")

                with st.expander(
                    f"{badge} — Topic: {item.get('topic_name')} | Score: {score_val:.1f}/10 | {timestamp}",
                    expanded=(idx == 1),
                ):
                    st.markdown(f"**Question ({item.get('difficulty')} | {item.get('category')}):**")
                    st.info(item.get("question_text", ""))

                    st.markdown("**Your Submitted Answer:**")
                    st.code(item.get("answer_text", ""), language="text")

                    col_h_left, col_h_right = st.columns(2)
                    with col_h_left:
                        st.markdown("**Verdict & Diagnostic:**")
                        st.write(f"- **Verdict:** `{item.get('verdict')}`")
                        st.write(f"- **Feedback:** {item.get('feedback') or 'No feedback recorded.'}")

                    with col_h_right:
                        if item.get("strengths"):
                            st.markdown("**Strengths:**")
                            for s in item["strengths"]:
                                st.markdown(f"- {s}")
                        if item.get("areas_for_improvement"):
                            st.markdown("**Areas for Improvement:**")
                            for a in item["areas_for_improvement"]:
                                st.markdown(f"- {a}")

                    if item.get("remediation_advice"):
                        st.warning(f"**Remediation Guidance:** {item['remediation_advice']}")
        else:
            st.info(
                f"No persistent evaluation history recorded for **{student_name}** yet.\n\n"
                "Go to the **'🎯 Adaptive Practice'** tab and submit an answer to test the full SQLite persistent storage cycle!",
                icon="ℹ️",
            )

    with tab_progress:
        st.subheader("📈 Current Preparation Progress")
        col_p1, col_p2 = st.columns(2)

        with col_p1:
            st.markdown("#### Completed Milestones")
            if daily_schedule:
                for item in daily_schedule:
                    d_num = item.get("day", 1)
                    done = d_num in st.session_state["completed_days"]
                    if done:
                        st.success(f"Day {d_num}: {item.get('focus')} — Completed")
                    else:
                        st.warning(f"Day {d_num}: {item.get('focus')} — Pending")
            else:
                st.info("No milestone data available.")

        with col_p2:
            st.markdown("#### Readiness Checklist")
            readiness = plan_structured.get("readiness_indicators", [])
            if readiness:
                for r in readiness:
                    st.checkbox(r, key=f"ready_{r[:25]}")
            else:
                st.markdown("- [ ] Solved 15+ standard placement questions")
                st.markdown("- [ ] Explained solutions clearly out loud")
                st.markdown("- [ ] Practiced edge cases & time complexity analysis")

        # ── Topic Mastery & Weakness Diagnostic Breakdown ────────────────────
        st.markdown("---")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("#### 🏆 Mastered Topics")
            if mastered_topics:
                for mt in mastered_topics:
                    st.success(f"✅ `{mt}` — Mastered (Score >= 7.0)")
            else:
                st.caption("No topics mastered yet. Complete practice questions with score >= 7.0 to achieve mastery.")

        with col_m2:
            st.markdown("#### ⚠️ Identified Weak Areas")
            if weak_topics:
                for wt in weak_topics:
                    st.error(f"⚠️ `{wt}` — Weakness Detected (Targeted Remediation Active)")
            else:
                st.caption("No weak topics currently flagged. Excellent work!")


    with tab_agent_log:
        st.subheader("🤖 Agent Reasoning & State")
        st.markdown("The LangGraph agent executed the following structured analysis:")

        st.json({
            "session_id": agent_data.get("session_id"),
            "student_profile": student_profile,
            "current_topic": current_topic,
            "recommended_next_task": recommended_next_task,
            "processing_time_seconds": agent_data.get("processing_time_seconds"),
        })

        if agent_data.get("messages"):
            st.markdown("#### Agent Dialogue Transcript")
            for msg in agent_data["messages"]:
                role = msg.get("role", "assistant")
                with st.chat_message("user" if role == "human" else "assistant"):
                    st.markdown(msg.get("content", ""))
