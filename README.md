# 🚀 EIT Preventive Intelligence

A real-time AI safety system that detects fatigue and predicts human risk using  
**EIT (Energy–Information Transformation)**.

---

## 🧠 Overview

This project combines:

- Computer vision for drowsiness detection
- EIT-based nonlinear risk modeling
- AI agent reasoning
- Session memory
- Real-time API
- Interactive dashboard

---

## ⚙️ Core Model

The system estimates:

- **Θ (Theta)** → structural stability  
- **Λ (Lambda)** → intelligence / reaction quality  
- **F** → physiological condition  
- **Load** → environmental pressure  

Then computes risk from these state variables.

---

## 📁 Main Files

- `eit_engine.py` — EIT scoring, risk, trend, alerts
- `cv_engine.py` — face/eye analysis and drowsiness scoring
- `ai_agent.py` — decision agent
- `agent_memory.py` — session memory
- `api.py` — FastAPI backend
- `dashboard.py` — Streamlit dashboard
- `train_alpha_beta.py` — parameter tuning
- `requirements.txt` — dependencies

---

## 🔥 Features

- Real-time fatigue-aware risk scoring
- Trend tracking over time
- AI-generated next-step recommendations
- Alert system
- Dashboard monitoring
- Expandable architecture for browser/mobile camera input

---

## 🚧 Status

**Current stage:** MVP / prototype

Planned next upgrades:

- live camera-to-backend integration
- stronger CV metrics
- deployment
- startup branding
- investor/demo version

---

## ▶️ Run Locally

### 1. Install dependencies
```bash
pip install -r requirements.txt
2. Train alpha and beta
python train_alpha_beta.py
3. Start the API
uvicorn api:app --reload
4. Start the dashboard
streamlit run dashboard.py
🌐 API Endpoints
GET / → API status
GET /health → health check
POST /predict → risk prediction
WS /ws/{session_id} → live streaming

## Authors

Toni Angel Mladenovski – Lead Developer  
Mihail Mladenski – Contributor
Built by eit-orbitos
