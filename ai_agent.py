from typing import Dict, Any


class AIAgent:
    def __init__(self):
        self.status_actions = {
            "GREEN": "Continue normal operation.",
            "YELLOW": "Monitor closely and reduce workload.",
            "ORANGE": "Prepare to stop and rest soon.",
            "RED": "Stop immediately and recover.",
        }

    def analyze(self, prediction: Dict[str, Any], trend: Dict[str, Any], camera: Dict[str, Any]) -> Dict[str, Any]:
        status = prediction.get("Status", "YELLOW")
        risk = prediction.get("Risk", 0.0)
        dominant = prediction.get("DominantFactor", "Unknown")
        trend_name = trend.get("Trend", "STABLE")

        cv = camera.get("cv_features", {}) if camera else {}
        drowsiness = cv.get("drowsiness", 0.0)
        closed_frames = cv.get("eyes_closed_frames", 0)

        reasoning = [
            f"Status = {status}",
            f"Risk = {risk}",
            f"Trend = {trend_name}",
            f"Dominant factor = {dominant}",
            f"Drowsiness = {drowsiness}",
            f"Eyes closed frames = {closed_frames}",
        ]

        escalation = False
        if trend_name in {"RISING", "RISING_FAST"}:
            escalation = True
        if drowsiness >= 0.75 or closed_frames >= 8:
            escalation = True

        action = self.status_actions.get(status, "Monitor closely.")
        if escalation and status in {"YELLOW", "ORANGE"}:
            action = "Escalate response: move to a safe stop as soon as possible."

        priority = {
            "GREEN": "low",
            "YELLOW": "medium",
            "ORANGE": "high",
            "RED": "critical",
        }.get(status, "medium")

        return {
            "agent_name": "EIT Safety Agent",
            "priority": priority,
            "message": f"Current system state is {status}.",
            "recommended_next_step": action,
            "reasoning": reasoning,
        }
