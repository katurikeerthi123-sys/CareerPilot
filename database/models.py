"""
CareerPilot — SQLAlchemy ORM Data Models (Phase 6 — Persistent Memory)

Defines models for:
  - User: Registered user accounts and authentication credentials
  - Student: Student profile and placement goals (linked to User)
  - Goal: Target technology, available days, and prep objectives
  - Topic: Curriculum topics with mastery and weakness flags
  - Question: Generated practice questions with hints and RAG context
  - Answer: Submitted answers from students
  - Evaluation: Detailed scoring, rubric verdict, strengths, and ideal answers
  - Progress: Aggregated performance metrics, topic mastery, and weak topics
  - StudySession: Persistent state between user visits and agent executions
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, List, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from database.connection import Base


class User(Base):
    """User account with authentication credentials and profile information."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(80), unique=True, index=True, nullable=False)
    email = Column(String(150), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(120), nullable=True)
    target_technology = Column(String(80), nullable=True, default="Java")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)

    # Relationships
    student = relationship("Student", back_populates="user", uselist=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "full_name": self.full_name or self.username,
            "target_technology": self.target_technology or "Java",
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None,
        }


class Student(Base):
    """Registered or active student preparing for placements."""
    __tablename__ = "students"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    name = Column(String(120), nullable=False, index=True)
    email = Column(String(120), nullable=True, unique=True, index=True)
    target_technology = Column(String(80), nullable=True, default="General")
    current_role = Column(String(80), nullable=True, default="Student / Candidate")
    available_days = Column(Integer, default=7)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="student")
    goals = relationship("Goal", back_populates="student", cascade="all, delete-orphan")
    progress_records = relationship("Progress", back_populates="student", cascade="all, delete-orphan")
    study_sessions = relationship("StudySession", back_populates="student", cascade="all, delete-orphan")
    answers = relationship("Answer", back_populates="student", cascade="all, delete-orphan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "name": self.name,
            "email": self.email,
            "target_technology": self.target_technology,
            "current_role": self.current_role,
            "available_days": self.available_days,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Goal(Base):
    """A placement preparation objective submitted by a student."""
    __tablename__ = "goals"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("students.id"), nullable=False, index=True)
    goal_text = Column(Text, nullable=False)
    target_technology = Column(String(80), nullable=True)
    days_available = Column(Integer, default=7)
    target_difficulty = Column(String(40), default="Intermediate")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    student = relationship("Student", back_populates="goals")
    topics = relationship("Topic", back_populates="goal", cascade="all, delete-orphan")
    progress = relationship("Progress", back_populates="goal", uselist=False, cascade="all, delete-orphan")
    study_sessions = relationship("StudySession", back_populates="goal", cascade="all, delete-orphan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "student_id": self.student_id,
            "goal_text": self.goal_text,
            "target_technology": self.target_technology,
            "days_available": self.days_available,
            "target_difficulty": self.target_difficulty,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Topic(Base):
    """An individual topic within a preparation curriculum."""
    __tablename__ = "topics"

    id = Column(Integer, primary_key=True, index=True)
    goal_id = Column(Integer, ForeignKey("goals.id"), nullable=False, index=True)
    name = Column(String(160), nullable=False, index=True)
    category = Column(String(80), nullable=True, default="Core")
    is_mastered = Column(Boolean, default=False)
    is_weak = Column(Boolean, default=False)
    order_index = Column(Integer, default=0)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    goal = relationship("Topic", back_populates="topics") if False else relationship("Goal", back_populates="topics")
    questions = relationship("Question", back_populates="topic", cascade="all, delete-orphan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "goal_id": self.goal_id,
            "name": self.name,
            "category": self.category,
            "is_mastered": self.is_mastered,
            "is_weak": self.is_weak,
            "order_index": self.order_index,
            "description": self.description,
        }


class Question(Base):
    """A practice question generated by the AI agent."""
    __tablename__ = "questions"

    id = Column(Integer, primary_key=True, index=True)
    topic_id = Column(Integer, ForeignKey("topics.id"), nullable=True, index=True)
    question_text = Column(Text, nullable=False)
    difficulty = Column(String(40), default="Medium")
    category = Column(String(80), default="Conceptual")
    is_remedial = Column(Boolean, default=False)
    hints_json = Column(Text, default="[]")
    ideal_answer = Column(Text, nullable=True)
    retrieved_context = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    topic = relationship("Topic", back_populates="questions")
    answers = relationship("Answer", back_populates="question", cascade="all, delete-orphan")

    @property
    def hints(self) -> list[str]:
        try:
            return json.loads(self.hints_json or "[]")
        except Exception:
            return []

    @hints.setter
    def hints(self, value: list[str]) -> None:
        self.hints_json = json.dumps(value or [])

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "topic_id": self.topic_id,
            "question_text": self.question_text,
            "difficulty": self.difficulty,
            "category": self.category,
            "is_remedial": self.is_remedial,
            "hints": self.hints,
            "ideal_answer": self.ideal_answer,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Answer(Base):
    """A student's submitted response to a question."""
    __tablename__ = "answers"

    id = Column(Integer, primary_key=True, index=True)
    question_id = Column(Integer, ForeignKey("questions.id"), nullable=False, index=True)
    student_id = Column(Integer, ForeignKey("students.id"), nullable=False, index=True)
    answer_text = Column(Text, nullable=False)
    time_taken_seconds = Column(Float, nullable=True)
    submitted_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    question = relationship("Question", back_populates="answers")
    student = relationship("Student", back_populates="answers")
    evaluation = relationship("Evaluation", back_populates="answer", uselist=False, cascade="all, delete-orphan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question_id": self.question_id,
            "student_id": self.student_id,
            "answer_text": self.answer_text,
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
        }


class Evaluation(Base):
    """Detailed grading and diagnostic feedback for a submitted answer."""
    __tablename__ = "evaluations"

    id = Column(Integer, primary_key=True, index=True)
    answer_id = Column(Integer, ForeignKey("answers.id"), unique=True, nullable=False, index=True)
    score = Column(Float, nullable=False)
    is_passed = Column(Boolean, default=False)
    verdict = Column(String(80), nullable=False)
    feedback = Column(Text, nullable=True)
    strengths_json = Column(Text, default="[]")
    areas_for_improvement_json = Column(Text, default="[]")
    remediation_advice = Column(Text, nullable=True)
    evaluated_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    answer = relationship("Answer", back_populates="evaluation")

    @property
    def strengths(self) -> list[str]:
        try:
            return json.loads(self.strengths_json or "[]")
        except Exception:
            return []

    @strengths.setter
    def strengths(self, value: list[str]) -> None:
        self.strengths_json = json.dumps(value or [])

    @property
    def areas_for_improvement(self) -> list[str]:
        try:
            return json.loads(self.areas_for_improvement_json or "[]")
        except Exception:
            return []

    @areas_for_improvement.setter
    def areas_for_improvement(self, value: list[str]) -> None:
        self.areas_for_improvement_json = json.dumps(value or [])

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "answer_id": self.answer_id,
            "score": self.score,
            "is_passed": self.is_passed,
            "verdict": self.verdict,
            "feedback": self.feedback,
            "strengths": self.strengths,
            "areas_for_improvement": self.areas_for_improvement,
            "remediation_advice": self.remediation_advice,
            "evaluated_at": self.evaluated_at.isoformat() if self.evaluated_at else None,
        }


class Progress(Base):
    """Aggregate progress and diagnostic tracker for a student on a specific goal."""
    __tablename__ = "progress"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("students.id"), nullable=False, index=True)
    goal_id = Column(Integer, ForeignKey("goals.id"), nullable=False, index=True)
    completed_topics_json = Column(Text, default="[]")
    weak_topics_json = Column(Text, default="[]")
    mastered_topics_json = Column(Text, default="[]")
    topic_scores_json = Column(Text, default="{}")
    mastered_count = Column(Integer, default=0)
    overall_progress_percent = Column(Integer, default=0)
    total_questions_attempted = Column(Integer, default=0)
    average_score = Column(Float, default=0.0)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    student = relationship("Student", back_populates="progress_records")
    goal = relationship("Goal", back_populates="progress")

    @property
    def completed_topics(self) -> list[str]:
        try:
            return json.loads(self.completed_topics_json or "[]")
        except Exception:
            return []

    @completed_topics.setter
    def completed_topics(self, val: list[str]) -> None:
        self.completed_topics_json = json.dumps(val or [])

    @property
    def weak_topics(self) -> list[str]:
        try:
            return json.loads(self.weak_topics_json or "[]")
        except Exception:
            return []

    @weak_topics.setter
    def weak_topics(self, val: list[str]) -> None:
        self.weak_topics_json = json.dumps(val or [])

    @property
    def mastered_topics(self) -> list[str]:
        try:
            return json.loads(self.mastered_topics_json or "[]")
        except Exception:
            return []

    @mastered_topics.setter
    def mastered_topics(self, val: list[str]) -> None:
        self.mastered_topics_json = json.dumps(val or [])

    @property
    def topic_scores(self) -> dict[str, list[float]]:
        try:
            return json.loads(self.topic_scores_json or "{}")
        except Exception:
            return {}

    @topic_scores.setter
    def topic_scores(self, val: dict[str, list[float]]) -> None:
        self.topic_scores_json = json.dumps(val or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "student_id": self.student_id,
            "goal_id": self.goal_id,
            "completed_topics": self.completed_topics,
            "weak_topics": self.weak_topics,
            "mastered_topics": self.mastered_topics,
            "topic_scores": self.topic_scores,
            "mastered_count": self.mastered_count,
            "overall_progress_percent": self.overall_progress_percent,
            "total_questions_attempted": self.total_questions_attempted,
            "average_score": round(self.average_score, 2),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class StudySession(Base):
    """A distinct study or preparation session preserving agent state across runs."""
    __tablename__ = "study_sessions"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("students.id"), nullable=False, index=True)
    goal_id = Column(Integer, ForeignKey("goals.id"), nullable=True, index=True)
    session_token = Column(String(100), unique=True, index=True, nullable=False)
    current_topic = Column(String(160), nullable=True)
    plan_markdown = Column(Text, nullable=True)
    plan_structured_json = Column(Text, default="{}")
    recommended_next_task = Column(Text, nullable=True)
    next_action = Column(String(60), default="continue")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    student = relationship("Student", back_populates="study_sessions")
    goal = relationship("Goal", back_populates="study_sessions")

    @property
    def plan_structured(self) -> dict[str, Any]:
        try:
            return json.loads(self.plan_structured_json or "{}")
        except Exception:
            return {}

    @plan_structured.setter
    def plan_structured(self, val: dict[str, Any]) -> None:
        self.plan_structured_json = json.dumps(val or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "student_id": self.student_id,
            "goal_id": self.goal_id,
            "session_token": self.session_token,
            "current_topic": self.current_topic,
            "plan_markdown": self.plan_markdown,
            "plan_structured": self.plan_structured,
            "recommended_next_task": self.recommended_next_task,
            "next_action": self.next_action,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
