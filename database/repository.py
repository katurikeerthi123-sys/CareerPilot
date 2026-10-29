"""
CareerPilot — Database Repository & CRUD Layer (Phase 6 — Persistent Memory)

Provides high-level functions to persist and retrieve:
  - Students and active preparation goals
  - Curriculum topics with mastery/weakness flags
  - AI questions, student answers, and detailed evaluations
  - Longitudinal progress and historical performance memory for agent decision-making
"""

from __future__ import annotations

import json
from typing import Any, Optional

from loguru import logger
from sqlalchemy import func
from sqlalchemy.orm import Session

from database.models import (
    Answer,
    Evaluation,
    Goal,
    Progress,
    Question,
    Student,
    StudySession,
    Topic,
    User,
)


# ── Student & Goal Management ─────────────────────────────────────────────────

def get_or_create_student(
    db: Session,
    name: str,
    target_technology: str = "General",
    available_days: int = 7,
    email: Optional[str] = None,
    user_id: Optional[int] = None,
) -> Student:
    """Find student by user_id, email, or name, or create a new student record."""
    clean_name = name.strip() or "Student"
    student = None

    # Priority 1: Match on user_id if provided
    if user_id is not None:
        student = db.query(Student).filter(Student.user_id == user_id).first()

    # Priority 2: Match on email if provided
    if not student and email:
        clean_email = email.strip().lower()
        student = db.query(Student).filter(func.lower(Student.email) == clean_email).first()

    # Priority 3: Match on name (fallback for unauthenticated or guest profiles)
    if not student:
        query = db.query(Student).filter(func.lower(Student.name) == clean_name.lower())
        if user_id is not None:
            query = query.filter((Student.user_id.is_(None)) | (Student.user_id == user_id))
        student = query.first()

    if not student:
        student = Student(
            user_id=user_id,
            name=clean_name,
            email=email.strip().lower() if email else None,
            target_technology=target_technology,
            available_days=available_days,
        )
        db.add(student)
        db.commit()
        db.refresh(student)
        logger.info(f"Created new persistent student profile: id={student.id}, user_id={student.user_id}, name='{student.name}'")
    else:
        # Update current settings and link user_id if not yet bound
        student.target_technology = target_technology
        student.available_days = available_days
        if user_id is not None and student.user_id is None:
            student.user_id = user_id
        if email and not student.email:
            student.email = email.strip().lower()
        db.commit()
        db.refresh(student)
        logger.debug(f"Retrieved existing student profile: id={student.id}, name='{student.name}'")

    return student


def get_or_create_goal(
    db: Session,
    student_id: int,
    goal_text: str,
    target_technology: str = "General",
    days_available: int = 7,
) -> Goal:
    """Find active goal or create a new preparation goal for a student."""
    goal = (
        db.query(Goal)
        .filter(Goal.student_id == student_id, Goal.is_active == True)
        .order_by(Goal.created_at.desc())
        .first()
    )

    if not goal:
        goal = Goal(
            student_id=student_id,
            goal_text=goal_text.strip(),
            target_technology=target_technology,
            days_available=days_available,
            is_active=True,
        )
        db.add(goal)
        db.commit()
        db.refresh(goal)
        logger.info(f"Created new goal: id={goal.id} for student_id={student_id}")
    else:
        goal.goal_text = goal_text.strip()
        goal.target_technology = target_technology
        goal.days_available = days_available
        db.commit()
        db.refresh(goal)

    return goal


# ── Topic & Curriculum Management ─────────────────────────────────────────────

def sync_goal_topics(
    db: Session,
    goal_id: int,
    topics_list: list[str],
) -> list[Topic]:
    """Ensure all topics from a generated curriculum exist in the database."""
    existing = {t.name.lower(): t for t in db.query(Topic).filter(Topic.goal_id == goal_id).all()}
    synced = []

    for idx, topic_name in enumerate(topics_list):
        clean_name = topic_name.strip()
        if not clean_name:
            continue
        key = clean_name.lower()
        if key in existing:
            synced.append(existing[key])
        else:
            new_topic = Topic(
                goal_id=goal_id,
                name=clean_name,
                order_index=idx,
                is_mastered=False,
                is_weak=False,
            )
            db.add(new_topic)
            synced.append(new_topic)

    db.commit()
    for t in synced:
        db.refresh(t)
    return synced


def find_or_create_topic(
    db: Session,
    goal_id: int,
    topic_name: str,
) -> Topic:
    """Find a specific topic under a goal or create it."""
    clean_name = topic_name.strip() or "General"
    topic = (
        db.query(Topic)
        .filter(Topic.goal_id == goal_id, func.lower(Topic.name) == clean_name.lower())
        .first()
    )
    if not topic:
        topic = Topic(goal_id=goal_id, name=clean_name)
        db.add(topic)
        db.commit()
        db.refresh(topic)
    return topic


# ── Study Sessions ────────────────────────────────────────────────────────────

def save_study_session(
    db: Session,
    session_id: str,
    student_id: int,
    goal_id: Optional[int],
    current_topic: Optional[str],
    plan_markdown: Optional[str],
    plan_structured: Optional[dict[str, Any]],
    recommended_next_task: Optional[str],
    next_action: str = "continue",
) -> StudySession:
    """Persist or update an active preparation session."""
    session = (
        db.query(StudySession)
        .filter(StudySession.session_token == session_id)
        .first()
    )

    if not session:
        session = StudySession(
            student_id=student_id,
            goal_id=goal_id,
            session_token=session_id,
            current_topic=current_topic,
            plan_markdown=plan_markdown,
            recommended_next_task=recommended_next_task,
            next_action=next_action,
        )
        if plan_structured:
            session.plan_structured = plan_structured
        db.add(session)
    else:
        session.current_topic = current_topic
        if plan_markdown:
            session.plan_markdown = plan_markdown
        if plan_structured:
            session.plan_structured = plan_structured
        if recommended_next_task:
            session.recommended_next_task = recommended_next_task
        session.next_action = next_action

    db.commit()
    db.refresh(session)
    return session


# ── Questions, Answers & Evaluations ──────────────────────────────────────────

def record_question(
    db: Session,
    goal_id: int,
    topic_name: str,
    question_text: str,
    difficulty: str = "Medium",
    category: str = "Conceptual",
    is_remedial: bool = False,
    hints: Optional[list[str]] = None,
    ideal_answer: Optional[str] = None,
    retrieved_context: Optional[str] = None,
) -> Question:
    """Save an AI-generated question linked to its topic."""
    topic = find_or_create_topic(db, goal_id, topic_name)
    question = Question(
        topic_id=topic.id,
        question_text=question_text.strip(),
        difficulty=difficulty,
        category=category,
        is_remedial=is_remedial,
        ideal_answer=ideal_answer,
        retrieved_context=retrieved_context,
    )
    if hints:
        question.hints = hints
    db.add(question)
    db.commit()
    db.refresh(question)
    return question


def record_answer_and_evaluation(
    db: Session,
    student_id: int,
    question_id: int,
    answer_text: str,
    eval_data: dict[str, Any],
    time_taken_seconds: Optional[float] = None,
) -> tuple[Answer, Evaluation]:
    """
    Store student's answer and the AI evaluation.
    Also updates Topic table flags (is_mastered, is_weak) according to score.
    """
    answer = Answer(
        question_id=question_id,
        student_id=student_id,
        answer_text=answer_text.strip(),
        time_taken_seconds=time_taken_seconds,
    )
    db.add(answer)
    db.commit()
    db.refresh(answer)

    score = float(eval_data.get("score", 0.0))
    is_passed = bool(eval_data.get("is_passed", score >= 7.0))
    verdict = eval_data.get("verdict", "Pass" if is_passed else "Weak")

    evaluation = Evaluation(
        answer_id=answer.id,
        score=score,
        is_passed=is_passed,
        verdict=verdict,
        feedback=eval_data.get("feedback", ""),
        remediation_advice=eval_data.get("remediation_advice"),
    )
    if eval_data.get("strengths"):
        evaluation.strengths = eval_data["strengths"]
    if eval_data.get("areas_for_improvement"):
        evaluation.areas_for_improvement = eval_data["areas_for_improvement"]

    db.add(evaluation)
    db.commit()
    db.refresh(evaluation)

    # Update associated Topic mastery / weakness flags
    q = db.query(Question).filter(Question.id == question_id).first()
    if q and q.topic_id:
        topic = db.query(Topic).filter(Topic.id == q.topic_id).first()
        if topic:
            if is_passed:
                topic.is_mastered = True
                topic.is_weak = False
            else:
                topic.is_mastered = False
                topic.is_weak = True
            db.commit()

    return answer, evaluation


# ── Progress Tracking ─────────────────────────────────────────────────────────

def update_progress(
    db: Session,
    student_id: int,
    goal_id: int,
    completed_topics: list[str],
    weak_topics: list[str],
    mastered_topics: list[str],
    topic_scores: dict[str, list[float]],
) -> Progress:
    """Compute and persist the student's aggregate progress."""
    prog = (
        db.query(Progress)
        .filter(Progress.student_id == student_id, Progress.goal_id == goal_id)
        .first()
    )

    if not prog:
        prog = Progress(
            student_id=student_id,
            goal_id=goal_id,
        )
        db.add(prog)

    prog.completed_topics = completed_topics
    prog.weak_topics = weak_topics
    prog.mastered_topics = mastered_topics
    prog.topic_scores = topic_scores
    prog.mastered_count = len(mastered_topics)

    # Compute questions count and average score from Answer & Evaluation tables
    total_answers = (
        db.query(Answer)
        .filter(Answer.student_id == student_id)
        .count()
    )
    avg_score = (
        db.query(func.avg(Evaluation.score))
        .join(Answer, Answer.id == Evaluation.answer_id)
        .filter(Answer.student_id == student_id)
        .scalar()
    ) or 0.0

    prog.total_questions_attempted = total_answers
    prog.average_score = float(avg_score)

    # Calculate overall progress percent based on mastered topics count vs curriculum topics
    total_topics = db.query(Topic).filter(Topic.goal_id == goal_id).count()
    if total_topics > 0:
        prog.overall_progress_percent = min(100, int((len(mastered_topics) / total_topics) * 100))
    elif completed_topics:
        total_attempted = max(len(completed_topics), len(mastered_topics))
        prog.overall_progress_percent = min(100, int((len(mastered_topics) / total_attempted) * 100))
    else:
        prog.overall_progress_percent = min(100, len(mastered_topics) * 20)

    db.commit()
    db.refresh(prog)
    return prog


# ── Persistent Memory for Agent ───────────────────────────────────────────────

def get_student_memory(
    db: Session,
    student_name: str,
    user_id: Optional[int] = None,
) -> dict[str, Any]:
    """
    Retrieve past performance memory for a student.
    Enables the Agent to recall weak topics, mastered topics, and historical scores
    across sessions. Matches on user_id if present, falling back to name.
    """
    student = None
    if user_id is not None:
        student = db.query(Student).filter(Student.user_id == user_id).first()

    if not student:
        clean_name = student_name.strip()
        student = db.query(Student).filter(func.lower(Student.name) == clean_name.lower()).first()

    if not student:
        return {
            "has_memory": False,
            "student_id": None,
            "weak_topics": [],
            "mastered_topics": [],
            "completed_topics": [],
            "topic_scores": {},
            "total_attempts": 0,
            "average_score": 0.0,
            "latest_session": None,
        }

    # Find latest active goal
    goal = (
        db.query(Goal)
        .filter(Goal.student_id == student.id, Goal.is_active == True)
        .order_by(Goal.created_at.desc())
        .first()
    )

    prog = None
    if goal:
        prog = db.query(Progress).filter(Progress.goal_id == goal.id).first()

    latest_session = (
        db.query(StudySession)
        .filter(StudySession.student_id == student.id)
        .order_by(StudySession.created_at.desc())
        .first()
    )

    weak = prog.weak_topics if prog else []
    mastered = prog.mastered_topics if prog else []
    completed = prog.completed_topics if prog else []
    scores = prog.topic_scores if prog else {}
    avg_score = prog.average_score if prog else 0.0
    attempts = prog.total_questions_attempted if prog else 0

    return {
        "has_memory": True,
        "student_id": student.id,
        "goal_id": goal.id if goal else None,
        "weak_topics": weak,
        "mastered_topics": mastered,
        "completed_topics": completed,
        "topic_scores": scores,
        "total_attempts": attempts,
        "average_score": avg_score,
        "latest_session": latest_session.to_dict() if latest_session else None,
    }


# ── User Account & Authentication Management ──────────────────────────────────

def create_user(
    db: Session,
    username: str,
    email: str,
    hashed_password: str,
    full_name: Optional[str] = None,
    target_technology: str = "Java",
) -> User:
    """Create a new registered user in the database."""
    clean_username = username.strip().lower()
    clean_email = email.strip().lower()
    clean_full_name = full_name.strip() if full_name else clean_username

    user = User(
        username=clean_username,
        email=clean_email,
        hashed_password=hashed_password,
        full_name=clean_full_name,
        target_technology=target_technology or "Java",
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    # Automatically ensure a corresponding Student record exists bound to user_id
    get_or_create_student(
        db=db,
        name=clean_full_name,
        target_technology=target_technology,
        email=clean_email,
        user_id=user.id,
    )

    logger.info(f"Registered new user account: id={user.id}, username='{user.username}'")
    return user


def get_user_by_username(db: Session, username: str) -> Optional[User]:
    """Look up user by case-insensitive username."""
    if not username:
        return None
    return db.query(User).filter(func.lower(User.username) == username.strip().lower()).first()


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Look up user by case-insensitive email address."""
    if not email:
        return None
    return db.query(User).filter(func.lower(User.email) == email.strip().lower()).first()


def get_user_by_identifier(db: Session, identifier: str) -> Optional[User]:
    """Look up user by username or email address."""
    clean_id = identifier.strip().lower()
    if not clean_id:
        return None
    return (
        db.query(User)
        .filter((func.lower(User.username) == clean_id) | (func.lower(User.email) == clean_id))
        .first()
    )


def update_user_last_login(db: Session, user_id: int) -> Optional[User]:
    """Update user's last_login timestamp."""
    from datetime import datetime
    user = db.query(User).filter(User.id == user_id).first()
    if user:
        user.last_login = datetime.utcnow()
        db.commit()
        db.refresh(user)
    return user


def list_users(db: Session) -> list[User]:
    """Return all registered users."""
    return db.query(User).order_by(User.created_at.desc()).all()
