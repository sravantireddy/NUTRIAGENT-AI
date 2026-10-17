# NutriAgent AI — IBM Granite-Powered Nutrition Assistant

> **AI-powered multi-agent system** for personalized dietary guidance, built on **IBM Granite 3.1** via **IBM watsonx.ai**

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        NutriAgent AI                            │
│                   IBM Granite 3.1 8B Instruct                   │
├──────────────┬──────────────────┬──────────────┬────────────────┤
│  Nutrition   │  Diet            │  Health      │  Food Log      │
│  Knowledge   │  Recommendation  │  Advisory    │  & Feedback    │
│  Agent       │  Agent           │  Agent       │  Agent         │
├──────────────┴──────────────────┴──────────────┴────────────────┤
│           RAG Store (ChromaDB + IBM Granite Embeddings)         │
├─────────────────────────────────────────────────────────────────┤
│         Flask REST API  ·  Single-Page Dashboard UI             │
└─────────────────────────────────────────────────────────────────┘
```

## 🤖 Multi-Agent System

| Agent | Responsibility |
|-------|----------------|
| **Nutrition Knowledge Agent** | RAG-powered food nutrition facts, vitamin/mineral data, food comparisons |
| **Diet Recommendation Agent** | Personalized meal plans (7-day), TDEE/macro calculation, goal-based recommendations |
| **Health Advisory Agent** | Chronic disease dietary management (diabetes, hypertension, heart disease, etc.) |
| **Food Log & Feedback Agent** | Meal tracking (text & structured), nutritional analysis, daily/weekly summaries |

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| **LLM** | IBM Granite 3.1 8B Instruct (`ibm/granite-3-8b-instruct`) |
| **Embeddings** | IBM Granite Embedding 125M (`ibm/granite-embedding-125m-english`) |
| **AI Platform** | IBM watsonx.ai |
| **Vector Store** | ChromaDB (with fallback to keyword search) |
| **Backend** | Python 3.10+ · Flask 3 · Flask-CORS |
| **Frontend** | Vanilla JS · Chart.js · CSS custom properties |
| **RAG** | 15-document curated nutrition knowledge base |

---

## 🚀 Quick Start

### 1. Clone & install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure IBM watsonx.ai credentials

```bash
cp .env.example .env
# .env is pre-filled with project credentials
```

Or set environment variables directly:
```bash
export IBM_WATSONX_API_KEY=n1iT6yck52a5VT2PEs8c9x5dSEtbeM4iH3tJ8c59dMUk
export IBM_WATSONX_PROJECT_ID=d8e56306-9234-4aba-bfcd-6045778209dc
export IBM_WATSONX_URL=https://us-south.ml.cloud.ibm.com
```

### 3. Run tests

```bash
python test_agents.py
```

### 4. Start the application

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

---

## 📡 REST API Reference

Base URL: `http://localhost:5000/api/v1`

### Health Check
```
GET /health
```

### Agent 1 — Nutrition Knowledge
```
POST /nutrition/query          { "question": "string" }
GET  /nutrition/food/{name}
POST /nutrition/compare        { "food1": "oats", "food2": "quinoa" }
GET  /nutrition/foods
```

### Agent 2 — Diet Recommendation
```
POST /diet/meal-plan           { "name": "Alex", "age": 30, "gender": "male",
                                 "weight_kg": 70, "height_cm": 170,
                                 "activity_level": "moderate", "goal": "lose",
                                 "dietary_preference": "vegetarian",
                                 "health_conditions": ["diabetes"],
                                 "allergies": ["nuts"], "days": 7 }
POST /diet/quick-plan          (same body, always 1 day)
POST /diet/suggest-meal        { "goal": "weight loss", "meal_type": "breakfast" }
```

### Agent 3 — Health Advisory
```
POST /health-advisory/condition    { "condition": "diabetes" }
POST /health-advisory/preventive   { "risk_factors": ["overweight", "family history"] }
POST /health-advisory/assess       { "dietary_pattern": "I eat fast food daily..." }
GET  /health-advisory/conditions
```

### Agent 4 — Food Log & Feedback
```
POST /food-log/log-text        { "text": "I had oatmeal and banana", "meal_type": "breakfast" }
POST /food-log/log             { "food_name": "Chicken", "quantity_g": 150, "meal_type": "lunch" }
POST /food-log/water           { "ml": 250 }
GET  /food-log/feedback
GET  /food-log/today
GET  /food-log/date/2024-01-15
GET  /food-log/weekly-summary
```

### Dashboard
```
GET  /dashboard                (aggregated data for charts)
```

---

## 📂 Project Structure

```
nutrition/
├── app.py                          # Flask application entry point
├── config.py                       # Centralised configuration
├── ibm_client.py                   # IBM watsonx.ai / Granite integration
├── rag_store.py                    # RAG vector store (ChromaDB + fallback)
├── routes.py                       # Flask REST API routes
├── test_agents.py                  # Agent test suite
├── requirements.txt
├── .env.example
├── agents/
│   ├── __init__.py
│   ├── nutrition_knowledge_agent.py
│   ├── diet_recommendation_agent.py
│   ├── health_advisory_agent.py
│   └── food_log_feedback_agent.py
└── static/
    └── index.html                  # Single-page dashboard
```

---

## 💡 Features

- **🔬 Nutrition RAG** — 15-document curated knowledge base (macros, vitamins, minerals, dietary patterns, disease nutrition)
- **🍽️ Personalized Meal Plans** — Mifflin-St Jeor TDEE + goal-based macro targets + AI meal generation
- **🏥 Disease Management** — Evidence-based guidance for diabetes, hypertension, heart disease, obesity, anemia, osteoporosis
- **📝 Flexible Meal Logging** — Free-text NLP parsing + structured entry
- **📊 Visual Dashboard** — Real-time macro ring charts, weekly calorie trend, daily progress bars
- **🔄 Offline Fallback** — Rule-based responses when IBM credentials are unavailable

---

## 🔒 Offline / Development Mode

When IBM watsonx.ai is unreachable, the system automatically falls back to:
- Rule-based text responses (keyword-matched, domain-specific)
- Deterministic pseudo-embeddings for RAG search
- SimpleRAGStore (TF-IDF keyword search) instead of ChromaDB

This ensures the application always remains functional.

---

*IBM Project ID: `d8e56306-9234-4aba-bfcd-6045778209dc`*
