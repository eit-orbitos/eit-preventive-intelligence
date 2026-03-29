from collections import deque
from typing import Dict, Any


class AgentMemory:
    def __init__(self, max_events: int = 100):
        self.max_events = max_events
        self.sessions = {}

    def add_event(self, session_id: str, prediction: Dict[str, Any], trend: Dict[str, Any], camera: Dict[str, Any]):
        if session_id not in self.sessions:
            self.sessions[session_id] = deque(maxlen=self.max_events)

        self.sessions[session_id].append({
            "prediction": prediction,
            "trend": trend,
            "camera": camera,
        })

    def get_summary(self, session_id: str) -> Dict[str, Any]:
        events = list(self.sessions.get(session_id, []))
        if not events:
            return {
                "count": 0,
                "avg_risk": 0.0,
                "peak_risk": 0.0,
                "red_count": 0,
                "orange_count": 0,
                "high_drowsiness_count": 0,
            }

        risks = [e["prediction"].get("Risk", 0.0) for e in events]
        statuses = [e["prediction"].get("Status", "GREEN") for e in events]
        drowsiness_vals = [
            (e.get("camera", {}).get("cv_features") or {}).get("drowsiness", 0.0)
            for e in events
        ]

        return {
            "count": len(events),
            "avg_risk": round(sum(risks) / len(risks), 4),
            "peak_risk": round(max(risks), 4),
            "red_count": sum(1 for s in statuses if s == "RED"),
            "orange_count": sum(1 for s in statuses if s == "ORANGE"),
            "high_drowsiness_count": sum(1 for d in drowsiness_vals if d >= 0.75),
        }
