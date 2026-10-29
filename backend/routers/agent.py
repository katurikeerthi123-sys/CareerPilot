"""
CareerPilot — Agent API Router (Phase 5 — Adaptive Learning Loop)

Exposes:
  - POST /api/agent           — Initialize preparation session, build curriculum plan, select Day 1 topic, generate Day 1 question
  - POST /api/agent/evaluate  — Evaluate student answer, adaptively detect weaknesses, update progress, decide next action (remediate or advance), and generate next question
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from agent.graph import graph
from database.connection import get_db
from database.repository import (
    get_or_create_goal,
    get_or_create_student,
    get_student_memory,
    record_answer_and_evaluation,
    record_question,
    save_study_session,
    sync_goal_topics,
    update_progress,
)

router = APIRouter(prefix="/api", tags=["Agent"])



# ── Request / Response schemas ────────────────────────────────────────────────

class AgentRequest(BaseModel):
    """Payload sent by the frontend to kick off a preparation session."""

    goal: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        examples=["I have a Java interview in 7 days. Prepare me."],
        description="The student's placement preparation goal in plain English.",
    )
    student_name: Optional[str] = Field(
        default=None,
        max_length=100,
        examples=["Arjun"],
        description="Optional student name for personalisation.",
    )
    target_technology: Optional[str] = Field(
        default=None,
        max_length=100,
        examples=["Java"],
        description="Optional target technology selected by the student.",
    )
    available_days: Optional[int] = Field(
        default=None,
        ge=1,
        le=90,
        examples=[7],
        description="Optional number of days available for preparation.",
    )
    user_id: Optional[int] = Field(
        default=None,
        description="Optional user ID for candidate profile linkage.",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="Optional Gemini API key passed from candidate session.",
    )


class DailyScheduleItem(BaseModel):
    day: int
    focus: str
    goals: list[str] = []
    topics: list[str] = []
    practice: str = ""
    estimated_hours: float = 2.0


class PlanStructured(BaseModel):
    plan_title: str = ""
    overview: str = ""
    daily_schedule: list[DailyScheduleItem] = []
    key_resources: list[str] = []
    success_tips: list[str] = []
    readiness_indicators: list[str] = []


class StudentProfile(BaseModel):
    technology: str = ""
    role: str = ""
    days_available: int = 7
    difficulty_level: str = "Intermediate"
    topics: list[str] = []
    interview_type: str = "Technical"
    summary: str = ""


class EvaluateRequest(BaseModel):
    """Payload sent when a student submits an answer to a question."""

    student_answer: str = Field(
        ...,
        min_length=1,
        examples=["HashMap uses an array of buckets and converts linked lists to Red-Black Trees when length >= 8."],
        description="Student's answer to the current practice question.",
    )
    current_question: Optional[str] = None
    question_details: Optional[Dict[str, Any]] = None
    current_topic: Optional[str] = None
    student_profile: Optional[Dict[str, Any]] = None
    weak_topics: Optional[List[str]] = None
    mastered_topics: Optional[List[str]] = None
    topic_scores: Optional[Dict[str, List[float]]] = None
    progress: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None
    user_id: Optional[int] = Field(
        default=None,
        description="Optional user ID for candidate profile linkage.",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="Optional Gemini API key passed from candidate session.",
    )


class AgentResponse(BaseModel):
    """Full agent response returned to the frontend."""

    success: bool
    session_id: Optional[str] = None
    student_name: Optional[str] = None
    student_profile: Optional[StudentProfile] = None
    plan_markdown: Optional[str] = None
    plan_structured: Optional[dict[str, Any]] = None
    current_topic: Optional[str] = None
    current_question: Optional[str] = None
    question_details: Optional[dict[str, Any]] = None
    evaluation: Optional[dict[str, Any]] = None
    weak_topics: list[str] = []
    mastered_topics: list[str] = []
    remediation_notes: Optional[str] = None
    recommended_next_task: Optional[str] = None
    progress: Optional[dict[str, Any]] = None
    next_action: Optional[str] = None
    messages: list[dict[str, str]] = []
    processing_time_seconds: float = 0.0
    error: Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/agent",
    response_model=AgentResponse,
    summary="Initialize CareerPilot session & generate plan with Day 1 question",
    description=(
        "Accepts a student's placement goal, runs LangGraph to analyze the goal, "
        "retrieves study materials from ChromaDB, constructs a plan, selects Day 1 topic, "
        "and serves the first practice question."
    ),
)
async def run_agent(
    request: AgentRequest,
    db: Session = Depends(get_db),
) -> AgentResponse:
    """
    Kicks off an adaptive preparation session.
    Retrieves student memory if previously prepared, executes the LangGraph curriculum workflow,
    and persists student, goal, topics, question, and progress into SQLite.
    """
    student_name = request.student_name or "Student"
    if request.api_key and request.api_key.strip():
        import os
        os.environ["GEMINI_API_KEY"] = request.api_key.strip()
        os.environ["GOOGLE_API_KEY"] = request.api_key.strip()

    logger.info(
        f"Agent initialization request | goal='{request.goal[:60]}…' | "
        f"student='{student_name}'"
    )

    start_time = time.perf_counter()

    # ── Compose goal with explicit overrides if provided ───────────────────
    goal_text = request.goal.strip()
    extra_details = []
    if request.target_technology and request.target_technology.lower() not in goal_text.lower():
        extra_details.append(f"Target technology: {request.target_technology}.")
    if request.available_days and str(request.available_days) not in goal_text:
        extra_details.append(f"Available preparation days: {request.available_days}.")
    if extra_details:
        goal_text = f"{goal_text} ({' '.join(extra_details)})"

    # ── Retrieve past persistent memory for this student ───────────────────
    try:
        past_mem = get_student_memory(db, student_name, user_id=request.user_id)
    except Exception as mem_err:
        logger.warning(f"Could not load student memory: {mem_err}")
        past_mem = {}

    prev_weak = list(past_mem.get("weak_topics") or [])
    prev_mastered = list(past_mem.get("mastered_topics") or [])
    prev_scores = dict(past_mem.get("topic_scores") or {})

    # ── Build initial state with persistent memory ──────────────────────────
    initial_state = {
        "student_goal": goal_text,
        "student_name": student_name,
        "weak_topics": prev_weak,
        "mastered_topics": prev_mastered,
        "topic_scores": prev_scores,
    }

    try:
        final_state = graph.invoke(initial_state)
        elapsed = round(time.perf_counter() - start_time, 2)
        logger.info(f"Agent workflow completed in {elapsed}s")

        if final_state.get("error"):
            logger.warning(f"Agent returned error: {final_state['error']}")
            return AgentResponse(
                success=False,
                error=final_state["error"],
                processing_time_seconds=elapsed,
            )

        # Extract plan
        plan_data = final_state.get("generated_plan") or {}
        plan_markdown = plan_data.get("markdown") if isinstance(plan_data, dict) else None
        plan_structured = plan_data.get("structured") if isinstance(plan_data, dict) else None

        # Extract profile
        profile_raw = final_state.get("student_profile") or {}
        if request.target_technology and profile_raw:
            profile_raw.setdefault("technology", request.target_technology)
        if request.available_days and profile_raw:
            profile_raw.setdefault("days_available", request.available_days)
        student_profile = StudentProfile(**profile_raw) if profile_raw else None

        # Extract progress and recommended task
        progress_data = final_state.get("progress") or {
            "completed_topics": [],
            "weak_topics": [],
            "session_count": 1,
            "overall_progress_percent": 0,
        }

        # ── Persist to SQLite Database ───────────────────────────────────────
        try:
            student = get_or_create_student(
                db=db,
                name=student_name,
                target_technology=request.target_technology or "General",
                available_days=request.available_days or 7,
                user_id=request.user_id,
            )
            goal = get_or_create_goal(
                db=db,
                student_id=student.id,
                goal_text=goal_text,
                target_technology=request.target_technology or "General",
                days_available=request.available_days or 7,
            )

            # Sync topics
            all_topics: list[str] = []
            if plan_structured and "daily_schedule" in plan_structured:
                for day in plan_structured.get("daily_schedule", []):
                    all_topics.extend(day.get("topics", []))
            if not all_topics and profile_raw and "topics" in profile_raw:
                all_topics = profile_raw.get("topics", [])
            if all_topics:
                sync_goal_topics(db, goal.id, all_topics)

            # Persist active study session
            save_study_session(
                db=db,
                session_id=final_state.get("session_id") or f"sess-{int(time.time())}",
                student_id=student.id,
                goal_id=goal.id,
                current_topic=final_state.get("current_topic"),
                plan_markdown=plan_markdown,
                plan_structured=plan_structured,
                recommended_next_task=final_state.get("recommended_next_task"),
                next_action=final_state.get("next_action") or "continue",
            )

            # Persist initial generated question if present
            if final_state.get("current_question"):
                q_det = final_state.get("question_details") or {}
                record_question(
                    db=db,
                    goal_id=goal.id,
                    topic_name=final_state.get("current_topic") or "General",
                    question_text=final_state["current_question"],
                    difficulty=q_det.get("difficulty", "Medium"),
                    category=q_det.get("category", "Conceptual"),
                    is_remedial=q_det.get("is_remedial", False),
                    hints=q_det.get("hints", []),
                    ideal_answer=q_det.get("ideal_answer"),
                )

            # Persist student progress
            update_progress(
                db=db,
                student_id=student.id,
                goal_id=goal.id,
                completed_topics=final_state.get("mastered_topics") or [],
                weak_topics=final_state.get("weak_topics") or [],
                mastered_topics=final_state.get("mastered_topics") or [],
                topic_scores=final_state.get("topic_scores") or {},
            )
        except Exception as persist_err:
            logger.warning(f"Failed to persist preparation session to SQLite: {persist_err}")

        return AgentResponse(
            success=True,
            session_id=final_state.get("session_id"),
            student_name=student_name,
            student_profile=student_profile,
            plan_markdown=plan_markdown,
            plan_structured=plan_structured,
            current_topic=final_state.get("current_topic"),
            current_question=final_state.get("current_question"),
            question_details=final_state.get("question_details"),
            weak_topics=final_state.get("weak_topics") or [],
            mastered_topics=final_state.get("mastered_topics") or [],
            recommended_next_task=final_state.get("recommended_next_task"),
            progress=progress_data,
            next_action=final_state.get("next_action"),
            messages=final_state.get("messages") or [],
            processing_time_seconds=elapsed,
            error=None,
        )


    except Exception as exc:
        elapsed = round(time.perf_counter() - start_time, 2)
        logger.exception(f"Unhandled error in agent workflow: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent workflow failed: {str(exc)}",
        )


@router.post(
    "/agent/evaluate",
    response_model=AgentResponse,
    summary="Submit student answer & execute adaptive learning cycle",
    description=(
        "Evaluates the student's answer, identifies weaknesses, updates progress, "
        "and adaptively decides whether to remediate or advance to the next topic."
    ),
)
async def evaluate_answer(
    request: EvaluateRequest,
    db: Session = Depends(get_db),
) -> AgentResponse:
    """
    Adaptive Evaluation Loop:
      Student Answer -> Evaluate -> Detect Weakness -> Update Progress -> Decide Next Action -> Generate Next Question
    Persists question, answer, evaluation scores, and updated topic mastery flags into SQLite.
    """
    if request.api_key and request.api_key.strip():
        import os
        os.environ["GEMINI_API_KEY"] = request.api_key.strip()
        os.environ["GOOGLE_API_KEY"] = request.api_key.strip()

    logger.info(f"Evaluation requested | topic='{request.current_topic}' | answer_len={len(request.student_answer)}")
    start_time = time.perf_counter()

    # Reconstruct state to feed into the LangGraph evaluation branch
    current_state = {
        "student_answer": request.student_answer.strip(),
        "current_question": request.current_question,
        "question_details": request.question_details or {},
        "current_topic": request.current_topic,
        "student_profile": request.student_profile or {},
        "weak_topics": request.weak_topics or [],
        "mastered_topics": request.mastered_topics or [],
        "topic_scores": request.topic_scores or {},
        "progress": request.progress or {},
        "session_id": request.session_id,
    }

    try:
        final_state = graph.invoke(current_state)
        elapsed = round(time.perf_counter() - start_time, 2)
        logger.info(f"Evaluation & adaptive cycle completed in {elapsed}s")

        if final_state.get("error"):
            return AgentResponse(
                success=False,
                error=final_state["error"],
                processing_time_seconds=elapsed,
            )

        profile_raw = final_state.get("student_profile") or {}
        student_profile = StudentProfile(**profile_raw) if profile_raw else None

        # ── Persist to SQLite Database ───────────────────────────────────────
        try:
            student_name = "Student"
            if request.student_profile and isinstance(request.student_profile, dict):
                student_name = (
                    request.student_profile.get("name")
                    or request.student_profile.get("student_name")
                    or "Student"
                )

            student = get_or_create_student(db=db, name=student_name, user_id=request.user_id)
            goal = get_or_create_goal(
                db=db,
                student_id=student.id,
                goal_text=f"Placement Preparation for {request.current_topic or 'General'}",
                target_technology=student.target_technology or "General",
            )

            # Record question
            q_det = request.question_details or {}
            question_record = record_question(
                db=db,
                goal_id=goal.id,
                topic_name=request.current_topic or "General",
                question_text=request.current_question,
                difficulty=q_det.get("difficulty", "Medium"),
                category=q_det.get("category", "Conceptual"),
                is_remedial=q_det.get("is_remedial", False),
                hints=q_det.get("hints", []),
            )

            # Record answer & evaluation
            if final_state.get("evaluation"):
                record_answer_and_evaluation(
                    db=db,
                    student_id=student.id,
                    question_id=question_record.id,
                    answer_text=request.student_answer,
                    eval_data=final_state["evaluation"],
                )

            # Persist updated progress
            completed_topics = (
                final_state.get("completed_topics")
                or (final_state.get("progress") or {}).get("completed_topics")
                or list((final_state.get("topic_scores") or {}).keys())
                or final_state.get("mastered_topics")
                or []
            )
            update_progress(
                db=db,
                student_id=student.id,
                goal_id=goal.id,
                completed_topics=completed_topics,
                weak_topics=final_state.get("weak_topics") or [],
                mastered_topics=final_state.get("mastered_topics") or [],
                topic_scores=final_state.get("topic_scores") or {},
            )

            # Persist new generated question if present and different from current
            if final_state.get("current_question") and final_state["current_question"] != request.current_question:
                next_q_det = final_state.get("question_details") or {}
                record_question(
                    db=db,
                    goal_id=goal.id,
                    topic_name=final_state.get("current_topic") or "General",
                    question_text=final_state["current_question"],
                    difficulty=next_q_det.get("difficulty", "Medium"),
                    category=next_q_det.get("category", "Conceptual"),
                    is_remedial=next_q_det.get("is_remedial", False),
                    hints=next_q_det.get("hints", []),
                    ideal_answer=next_q_det.get("ideal_answer"),
                )

            # Update session state
            if request.session_id:
                save_study_session(
                    db=db,
                    session_id=request.session_id,
                    student_id=student.id,
                    goal_id=goal.id,
                    current_topic=final_state.get("current_topic"),
                    plan_markdown=None,
                    plan_structured=None,
                    recommended_next_task=final_state.get("recommended_next_task"),
                    next_action=final_state.get("next_action") or "continue",
                )
        except Exception as persist_err:
            logger.warning(f"Failed to persist evaluation to SQLite: {persist_err}")

        return AgentResponse(
            success=True,
            session_id=final_state.get("session_id"),
            student_profile=student_profile,
            current_topic=final_state.get("current_topic"),
            current_question=final_state.get("current_question"),
            question_details=final_state.get("question_details"),
            evaluation=final_state.get("evaluation"),
            weak_topics=final_state.get("weak_topics") or [],
            mastered_topics=final_state.get("mastered_topics") or [],
            remediation_notes=final_state.get("remediation_notes"),
            recommended_next_task=final_state.get("recommended_next_task"),
            progress=final_state.get("progress"),
            next_action=final_state.get("next_action"),
            messages=final_state.get("messages") or [],
            processing_time_seconds=elapsed,
            error=None,
        )


    except Exception as exc:
        elapsed = round(time.perf_counter() - start_time, 2)
        logger.exception(f"Unhandled error in evaluation workflow: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation cycle failed: {str(exc)}",
        )
