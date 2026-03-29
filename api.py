from typing import Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from eit_engine import compute_risk, RiskTrendBuffer, build_alerts
from cv_engine import CVEngine
from ai_agent import AIAgent
from agent_memory import AgentMemory


app = FastAPI(title="EIT Preventive Intelligence API")

cv_engine = CVEngine()
ai_agent = AIAgent()
agent_memory = AgentMemory()
trend_buffers: Dict[str, RiskTrendBuffer] = {}


def get_trend_buffer(session_id: str) -> RiskTrendBuffer:
    if session_id not in trend_buffers:
        trend_buffers[session_id] = RiskTrendBuffer(maxlen=120)
    return trend_buffers[session_id]


class PredictInput(BaseModel):
    session_id: str = "default"
    hours_awake: float
    sleep_hours: float
    eye_strain: float
    reaction_time: float
    hydration: float
    heart_rate: float
    steering_variability: float
    lane_drift: float
    time_of_day: float
    food_intake: float


@app.get("/")
def root():
    return {"message": "EIT Preventive Intelligence API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: PredictInput):
    session_id = data.session_id

    features = {
        "hours_awake": data.hours_awake,
        "sleep_hours": data.sleep_hours,
        "eye_strain": data.eye_strain,
        "reaction_time": data.reaction_time,
        "hydration": data.hydration,
        "heart_rate": data.heart_rate,
        "steering_variability": data.steering_variability,
        "lane_drift": data.lane_drift,
        "time_of_day": data.time_of_day,
        "food_intake": data.food_intake,
    }

    prediction = compute_risk(features)
    trend = get_trend_buffer(session_id).update(prediction["Risk"])
    camera = {"cv_features": {"face_detected": False, "drowsiness": 0.0, "eyes_closed_frames": 0}}
    alerts = build_alerts(prediction, trend, camera)

    agent_memory.add_event(session_id, prediction, trend, camera)
    memory_summary = agent_memory.get_summary(session_id)

    agent_output = ai_agent.analyze(prediction, trend, camera)
    agent_output["memory_summary"] = memory_summary

    return {
        "prediction": prediction,
        "trend": trend,
        "camera": camera,
        "alerts": alerts,
        "agent": agent_output,
    }


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    trend_buffer = get_trend_buffer(session_id)

    try:
        while True:
            data = await websocket.receive_json()

            features = {
                "hours_awake": float(data.get("hours_awake", 0)),
                "sleep_hours": float(data.get("sleep_hours", 0)),
                "eye_strain": float(data.get("eye_strain", 0)),
                "reaction_time": float(data.get("reaction_time", 0)),
                "hydration": float(data.get("hydration", 0)),
                "heart_rate": float(data.get("heart_rate", 0)),
                "steering_variability": float(data.get("steering_variability", 0)),
                "lane_drift": float(data.get("lane_drift", 0)),
                "time_of_day": float(data.get("time_of_day", 0)),
                "food_intake": float(data.get("food_intake", 0)),
            }

            prediction = compute_risk(features)
            trend = trend_buffer.update(prediction["Risk"])

            camera = {
                "cv_features": {
                    "face_detected": False,
                    "drowsiness": 0.0,
                    "eyes_closed_frames": 0,
                }
            }

            alerts = build_alerts(prediction, trend, camera)

            agent_memory.add_event(session_id, prediction, trend, camera)
            memory_summary = agent_memory.get_summary(session_id)

            agent_output = ai_agent.analyze(prediction, trend, camera)
            agent_output["memory_summary"] = memory_summary

            await websocket.send_json({
                "prediction": prediction,
                "trend": trend,
                "camera": camera,
                "alerts": alerts,
                "agent": agent_output,
            })

    except WebSocketDisconnect:
        print(f"Client disconnected: {session_id}")
