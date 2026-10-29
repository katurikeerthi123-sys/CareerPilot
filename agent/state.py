"""
CareerPilot — LangGraph Agent State (Phase 1 skeleton)

TypedDict used as the canonical state schema passed between every node
in the LangGraph workflow. All fields are optional so nodes can be added
incrementally across phases without breaking existing ones.
"""

from __future__ import annotations

from typing import Any, Optional
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    """
    Shared state object for the CareerPilot LangGraph workflow.

    Fields
    ------
    student_goal : str
        The raw goal string provided by the student.
        e.g. "I have a Java interview in 7 days. Prepare me."

    student_name : str
        Name of the student (populated after login/profile step).

    user_id : int
        Authenticated user ID if available.

    student_profile : dict
        Structured profile: name, experience level, target role, etc.

    current_topic : str
        The topic currently being studied / questioned.

    messages : list[dict]
        Conversation history in LangChain message format
        [{"role": "human"|"ai", "content": "..."}].

    generated_plan : str | dict
        The preparation plan produced by the planning node.

    current_question : str
        The most recently generated practice question.

    student_answer : str
        The student's answer to the current question.

    evaluation : dict
        Result of evaluating the student's answer.
        e.g. {"score": 8, "feedback": "...", "correct": True}

    weak_topics : list[str]
        Topics where the student has scored below threshold.

    mastered_topics : list[str]
        Topics where the student has scored >= passing threshold.

    completed_topics : list[str]
        Topics where the student has completed at least one practice question.

    progress : dict
        Aggregate progress data:
        {"completed_topics": [...], "scores": {...}, "session_count": int}

    retrieved_context : str
        Text chunks retrieved from ChromaDB for the current topic/question.

    next_action : str
        Decision made by the adaptive node:
        "continue" | "repeat_topic" | "finish"

    recommended_next_task : str
        The immediate next actionable task recommended for the student.

    error : str
        Any error message to surface back to the API caller.

    topic_scores : dict[str, list[float]]
        Score history mapped per topic for longitudinal weakness tracking.

    question_details : dict
        Structured question payload (question, topic, expected_points, difficulty).

    remediation_notes : str
        Specific tips or document excerpts recommended to fix an identified weakness.

    session_id : str
        Persistent session identifier.
    """

    student_goal: Optional[str]
    student_name: Optional[str]
    user_id: Optional[int]
    student_profile: Optional[dict[str, Any]]
    current_topic: Optional[str]
    messages: Optional[list[dict[str, str]]]
    generated_plan: Optional[Any]
    current_question: Optional[str]
    student_answer: Optional[str]
    evaluation: Optional[dict[str, Any]]
    weak_topics: Optional[list[str]]
    mastered_topics: Optional[list[str]]
    completed_topics: Optional[list[str]]
    topic_scores: Optional[dict[str, list[float]]]
    question_details: Optional[dict[str, Any]]
    remediation_notes: Optional[str]
    progress: Optional[dict[str, Any]]
    retrieved_context: Optional[str]
    next_action: Optional[str]
    recommended_next_task: Optional[str]
    error: Optional[str]
    session_id: Optional[str]
