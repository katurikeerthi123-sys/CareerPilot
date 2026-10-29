"""
CareerPilot — Weakness Detector & Progress Updater Node (Phase 5)

Updates student knowledge state and progress:
  - Tracks score history per topic (state["topic_scores"])
  - If score < 7.0: marks topic as weak, removes from mastered, stores remediation advice
  - If score >= 7.0: marks topic as mastered, clears from weak list
  - Sets completed_topics as all topics attempted (scores recorded)
  - Recalculates aggregate readiness progress percentages based on curriculum topics
"""

from __future__ import annotations

from typing import Any, Dict, List
from loguru import logger

from agent.state import AgentState


def detect_weakness_and_update_progress(state: AgentState) -> AgentState:
    """
    LangGraph node: Analyze performance score, update weak/mastered topic sets,
    and update longitudinal progress.

    Reads:  state["evaluation"], state["current_topic"], state["weak_topics"], state["mastered_topics"]
    Writes: state["weak_topics"], state["mastered_topics"], state["completed_topics"], state["topic_scores"], state["progress"], state["remediation_notes"]
    """
    if state.get("error"):
        return state

    eval_data: Dict[str, Any] = state.get("evaluation") or {}
    score = float(eval_data.get("score", 0.0))
    topic = state.get("current_topic") or "General Technical"
    profile = state.get("student_profile") or {}
    all_curriculum_topics: List[str] = list(profile.get("topics", []))
    if topic not in all_curriculum_topics and all_curriculum_topics:
        all_curriculum_topics.append(topic)

    # 1. Update topic score history
    topic_scores: Dict[str, List[float]] = dict(state.get("topic_scores") or {})
    if topic not in topic_scores:
        topic_scores[topic] = []
    topic_scores[topic].append(score)

    weak_topics: List[str] = list(state.get("weak_topics") or [])
    mastered_topics: List[str] = list(state.get("mastered_topics") or [])
    remediation_notes: str = state.get("remediation_notes") or ""

    # 2. Adaptive weakness detection
    if score < 7.0:
        # Performance was substandard → flag topic as weak
        if topic not in weak_topics:
            weak_topics.append(topic)
        if topic in mastered_topics:
            mastered_topics.remove(topic)

        remedial_advice = eval_data.get("remediation_advice") or eval_data.get("feedback")
        remediation_notes = f"⚠️ Weakness identified in '{topic}': {remedial_advice}"
        logger.warning(f"Student performed below threshold ({score}/10) in '{topic}'. Flagged as WEAK.")

    else:
        # Performance was satisfactory or mastered → clear weak status
        if topic in weak_topics:
            weak_topics.remove(topic)
            logger.success(f"Weakness resolved for '{topic}'! Score: {score}/10.")
        if topic not in mastered_topics:
            mastered_topics.append(topic)
            logger.success(f"Topic '{topic}' mastered! Score: {score}/10.")

        remediation_notes = f"✅ Demonstrated mastery in '{topic}'."

    # 3. Explicit completed topics (all topics with at least one question attempt)
    completed_topics: List[str] = [t for t, scores in topic_scores.items() if scores]

    # 4. Aggregate progress calculation
    total_topics_count = max(len(all_curriculum_topics), len(completed_topics), 1)
    progress_pct = int((len(mastered_topics) / total_topics_count) * 100)

    # Calculate average score across all attempts
    all_scores = [s for score_list in topic_scores.values() for s in score_list]
    avg_score = round(sum(all_scores) / len(all_scores), 1) if all_scores else 0.0

    current_progress = dict(state.get("progress") or {})
    current_progress.update({
        "completed_topics": completed_topics,
        "weak_topics": weak_topics,
        "mastered_topics": mastered_topics,
        "mastered_count": len(mastered_topics),
        "total_topics": total_topics_count,
        "overall_progress_percent": min(progress_pct, 100),
        "total_questions_attempted": len(all_scores),
        "average_score": avg_score,
        "topic_scores": topic_scores,
    })

    return {
        **state,
        "completed_topics": completed_topics,
        "weak_topics": weak_topics,
        "mastered_topics": mastered_topics,
        "topic_scores": topic_scores,
        "progress": current_progress,
        "remediation_notes": remediation_notes,
    }
