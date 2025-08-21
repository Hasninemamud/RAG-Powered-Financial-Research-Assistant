"""
Minimal conversation memory for follow-ups.
Keeps last topic and last user question per session.
"""
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class SessionState:
    last_topic: str = ""
    last_user_question: str = ""
    history: List[str] = field(default_factory=list)


class ConversationMemory:
    def __init__(self):
        self.sessions: Dict[str, SessionState] = {}

    def _get(self, session_id: str) -> SessionState:
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionState()
        return self.sessions[session_id]

    def update(self, session_id: str, user_question: str, topic: str):
        s = self._get(session_id)
        s.history.append(user_question)
        s.last_user_question = user_question
        if topic:
            s.last_topic = topic

    def contextualize(self, session_id: str, user_question: str) -> str:
        """
        Heuristic: if very short/ambiguous or contains 'it/that/they/this', append last topic.
        """
        s = self._get(session_id)
        q = user_question.strip()
        tokens = q.lower().split()
        pronouns = {"it", "that", "they", "this", "those", "them", "there", "here"}
        if len(tokens) <= 5 or any(t in pronouns for t in tokens):
            if s.last_topic:
                q = f"{q} (context: {s.last_topic})"
        return q

    def infer_topic(self, user_question: str) -> str:
        """
        Very naive topic guess: pick frequent policy keywords.
        """
        kws = ["budget", "debt", "infrastructure", "tax", "revenue", "expenditure", "loan", "deficit", "grant", "policy"]
        ql = user_question.lower()
        for k in kws:
            if k in ql:
                return k
        # default: first 3 informative words
        return " ".join([w for w in ql.split() if len(w) > 3][:3])