# 🚀 CareerPilot: An Agentic AI for Personalized Placement Preparation

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![Gemini](https://img.shields.io/badge/Google%20Gemini-3.5%20Flash%20Lite-4285F4.svg?logo=google)](https://ai.google.dev/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-blue.svg)](https://www.trychroma.com/)
[![SQLite](https://img.shields.io/badge/SQLite-SQLAlchemy%20Persistence-003B57.svg?logo=sqlite)](https://www.sqlite.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20UI-FF4B4B.svg?logo=streamlit)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Compose%20Ready-2496ED.svg?logo=docker)](https://www.docker.com/)
[![CI](https://github.com/katurikeerthi123-sys/CareerPilot/actions/workflows/ci.yml/badge.svg)](https://github.com/katurikeerthi123-sys/CareerPilot/actions)

> **CareerPilot** is a genuinely agentic AI system designed for final-year college students preparing for technical placement drives. Unlike traditional chatbots or superficial RAG wrappers that simply return static answers, CareerPilot maintains a continuous **adaptive feedback loop**:
>
> **Goal → Analyze → Plan → Retrieve → Teach → Evaluate → Diagnose Weakness → Adapt → Re-evaluate**

---

## 🌟 Key Differentiator: Agentic vs. Chatbot

| Feature | Traditional Placement Chatbot | CareerPilot (Agentic Placement AI) |
|---|---|---|
| **Interaction Model** | User prompts → Static LLM response | Student declares goal → AI orchestrates multi-day preparation |
| **Workflow Engine** | Single-turn API call | Multi-node StateGraph built with **LangGraph** |
| **Adaptive Learning** | None; answers in isolation | Identifies weak topics (< 7.0), schedules remediation, re-tests |
| **Study Grounding** | Generic public web data | Semantic RAG with **ChromaDB** using curated placement materials |
| **Persistent Memory** | Lost upon session refresh | **SQLite** relational memory recalls past attempts and mastery |
| **Interview Simulation** | Unstructured Q&A | Multi-turn **Staff Interviewer Agent** with tech & comm grading |
| **Readiness Scoring** | Subjective / none | Quantitative **Readiness Score %** with hiring company tier matching |

---

## 🏗️ Architecture

```
                                    +--------------------+
                                    |     Student UI     |
                                    | (Streamlit Portal) |
                                    +---------+----------+
                                              | HTTP REST
                                              v
+------------------------------------------------------------------------------------------+
|                                    FastAPI Backend                                       |
|  /api/agent  |  /api/documents  |  /api/students  |  /api/placement  |  /api/interview   |
+---------------------------------------------+--------------------------------------------+
                                              |
                                              v
+------------------------------------------------------------------------------------------+
|                              LangGraph Agentic Ecosystem                                 |
|                                                                                          |
|  [Agentic Learning Loop]                                  [Mock Interview Agent]        |
|  START ──► Analyze Goal ──► Generate Plan                 START ──► Staff Interviewer   |
|     │                                                        │             ▲             |
|     ▼                                                        ▼             │             |
|  Select Topic ◄─── [Adaptive Decision]                   Candidate Ans    [Loop Next Q]  |
|     │                     ▲                                  │             │             |
|     ▼                     │                                  ▼             │             |
|  ChromaDB RAG ──► Evaluator Node                          Evaluator Agent ─┘             |
|  Context Retrieval        ▲ (Score Tech & Comm)              │                           |
|     │                     │                                  ▼                           |
|     ▼                     │                       Hiring Committee Scorecard             |
|  Generate Question ───────┘                       (Verdict, Tech/Comm Scores, Actions)   |
+------------------------------------------------------------------------------------------+
           |                                                      |
           v                                                      v
+----------------------+                               +----------------------+
| ChromaDB Vectorstore |                               |   SQLite Database    |
| (10 Placement Guides |                               | (Student, Progress,  |
|  44 Semantic Chunks) |                               |  Question, Answer,   |
+----------------------+                               |  Evaluation, Memory) |
                                                       +----------------------+
```

---

## 🚀 Features

### 1. Personalized Study Planning
- Generates custom day-by-day preparation schedules based on student goals, available days, and target technologies (Java, Python, C++, Web, etc.).

### 2. Semantic Placement RAG (ChromaDB)
- Ingests official placement syllabus and interview guides:
  - **Accenture**, **Cognizant**, **Google**, **Infosys**, **TCS NQT**, **Mock Interview Guides**, **Resume Guidelines**, and **Java Core Placement Manuals**.
- Grounded contextual retrieval with Gemini embeddings (`models/gemini-embedding-001`).

### 3. Adaptive Learning & Weakness Remediation Loop
- Evaluates student answers against technical correctness benchmarks.
- Diagnoses weaknesses (< 7.0/10.0 threshold) and dynamically updates candidate's mastery state.
- Automatically adjusts next curriculum steps: repeats weak topics with remedial questions or advances upon mastery.

### 4. Multi-Format Placement Practice
- **🔘 Multiple Choice Questions (MCQ):** Practice questions with instant answer verification, options analysis, and explanations.
- **💻 Coding Challenges:** Problem statements with sample test cases, constraints, starter code skeletons, and optimal complexity guidelines.
- **🎯 Conceptual Deep Dives:** Deep technical questions probing runtime semantics, memory models, and internal working.

### 5. Final Placement Readiness Report
- Synthesizes student's complete historical record into an executive scorecard:
  - **Overall Readiness Score (0-100%)**
  - **Company-Fit Tier Matching** (Tier-1 Product, Unicorn Startups, IT Services MNCs)
  - **Topic Competency Matrix** with status tags and interview recommendations
  - **Prioritized 3-Step Action Plan**

### 6. Interactive Mock Interview Agent
- Poses progressive interview turns from foundational concepts to system edge cases.
- Evaluates dual dimensions:
  - **Technical Correctness & Depth (0-10)**
  - **Communication Clarity & Articulation (0-10)**
- Delivers a final **Lead Hiring Committee Decision** (`Strong Hire`, `Hire`, `Leaning Hire`, `No Hire`) with actionable feedback.

---

## 🛠️ Tech Stack

- **Language:** Python 3.11, 3.12 (Tested and supported in CI)
- **Agent Framework:** LangGraph 0.2+ & LangChain
- **LLM Engine:** Google Gemini (Gemini 3.5 Flash Lite) via `langchain-google-genai`
- **Backend API:** FastAPI & Uvicorn
- **Vector Database:** ChromaDB
- **Relational Storage:** SQLite with SQLAlchemy ORM
- **Frontend Dashboard:** Streamlit
- **Testing:** Pytest (95 automated tests)

---

## 📦 Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/katurikeerthi123-sys/CareerPilot.git
cd CareerPilot
```

### 2. Create and activate a virtual environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Open `.env` and set your Google Gemini API Key:
```ini
# Core AI Credentials
GEMINI_API_KEY="your-gemini-api-key"
LLM_MODEL="gemini-3.5-flash-lite"

# Server & Security
APP_ENV="development"                  # Set to "production" for strict startup checks
APP_HOST="0.0.0.0"
APP_PORT=8000
CORS_ALLOWED_ORIGINS='["http://localhost:8501","http://127.0.0.1:8501"]'
AUTH_SECRET_KEY="your-cryptographic-secret-key"
TOKEN_EXPIRY_HOURS=168
```

> **Production Security Note:** When `APP_ENV=production`, the application performs strict startup validation: `GEMINI_API_KEY` must be set, and wildcard `*` CORS origins are rejected.


---

## 🏃 Running the Application

### Start the FastAPI Backend Server
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
API Documentation will be live at:
- **Interactive Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)

### Start the Streamlit Frontend Dashboard
In a second terminal:
```bash
streamlit run frontend/app.py --server.port 8501
```
Open your browser to: **[http://localhost:8501](http://localhost:8501)**

---

## 🐳 Docker Deployment (Recommended)

Launch both the **FastAPI Backend** and the **Streamlit Frontend** with persistent storage in one command:

```bash
# 1. Set your Gemini API key in .env or your terminal
export GEMINI_API_KEY="your-gemini-api-key-here"

# 2. Build and launch services in detached mode
docker compose up --build -d

# 3. Check container health status
docker compose ps
```

- **Candidate Web Dashboard:** [http://localhost:8501](http://localhost:8501)
- **API Swagger Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Backend Health Check:** [http://localhost:8000/health](http://localhost:8000/health)

To view streaming logs or shut down containers:
```bash
docker compose logs -f              # View real-time logs
docker compose down                 # Stop services and preserve storage
```
For complete production options, volume mounting, and backup commands, see [`DOCKER_GUIDE.md`](./DOCKER_GUIDE.md).

---

## 🧪 Running Automated Tests

CareerPilot includes an exhaustive test suite covering all phases (100% passing across 126 tests):

```bash
# Run all unit and integration tests (126 tests)
pytest -v -m "not live"

# Run tests by module / phase
pytest tests/test_phase5.py -v              # Agentic Learning Loop
pytest tests/test_phase6.py -v              # SQLite Persistent Memory
pytest tests/test_phase7.py -v              # Placement MCQs, Coding & Readiness
pytest tests/test_phase8.py -v              # Mock Interview Agent & Scorecard
pytest tests/test_phase9_comprehensive.py -v # Full API & State Boundary Suite
pytest tests/test_phase10.py -v             # Docker & Compose Configurations
```

---

## 📂 Project Structure

```
CareerPilot/
│
├── backend/                     # FastAPI Application Layer
│   ├── main.py                  # API entry point & lifespan
│   ├── config.py                # Pydantic Settings
│   └── routers/                 # Modular API Routers
│       ├── agent.py             # Agentic loop endpoints
│       ├── documents.py         # Document ingestion & RAG search
│       ├── interview.py         # Mock interview endpoints
│       ├── placement.py         # Multi-format questions & readiness
│       └── students.py          # Student memory & history endpoints
│
├── agent/                       # Core LangGraph Learning Loop
│   ├── graph.py                 # Compiled StateGraph workflow
│   ├── state.py                 # AgentState TypedDict
│   ├── llm.py                   # Gemini LLM factory
│   ├── prompts.py               # Evaluator & planner prompts
│   └── nodes/                   # Specialized LangGraph nodes
│       ├── goal_analyzer.py
│       ├── planner.py
│       ├── topic_selector.py
│       ├── retriever.py
│       ├── question_generator.py
│       ├── evaluator.py
│       ├── weakness_detector.py
│       └── adaptive_decider.py
│
├── interview/                   # Mock Interview Agent System
│   ├── prompts.py               # Interviewer & evaluator system prompts
│   ├── schemas.py               # Turn & report models
│   ├── service.py               # Multi-turn session manager
│   └── workflow.py              # LangGraph interview StateGraph
│
├── placement/                   # Placement Testing Features
│   ├── generator.py             # MCQ & coding challenge generator
│   └── schemas.py               # Question formats & readiness models
│
├── rag/                         # Document Ingestion & Vector Retrieval
│   ├── embeddings.py            # Gemini vector embeddings
│   ├── ingestion.py             # Chunking & loader engine
│   ├── retriever.py             # Semantic RAG context search
│   └── vectorstore.py           # ChromaDB client interface
│
├── database/                    # Persistent Relational Storage
│   ├── connection.py            # SQLite engine & sessionmaker
│   ├── models.py                # SQLAlchemy ORM models
│   └── repository.py            # CRUD repository operations
│
├── frontend/                    # Streamlit Dashboard
│   └── app.py                   # Multi-tab interactive UI
│
├── data/
│   └── documents/               # Curated placement study guides
│
├── tests/                       # Complete Test Harness (Phases 1-10)
│   ├── conftest.py              # Shared fixtures & DB in-memory engine
│   ├── sample_data.py           # Test student profiles & answer sets
│   ├── test_phase1.py to test_phase8.py
│   ├── test_phase9_comprehensive.py # End-to-end integration suite
│   └── test_phase10.py          # Docker & deployment validation
│
├── Dockerfile                   # Multi-service production container image
├── docker-compose.yml           # Orchestration for backend & frontend
├── .dockerignore                # Clean build exclusions
├── DOCKER_GUIDE.md              # Container deployment guide
├── CONTRIBUTING.md              # Developer contribution guidelines
├── LICENSE                      # MIT Open Source License
├── .env.example                 # Secrets template
├── .gitignore                   # Safe exclusions (.env, *.db, chroma_db/)
├── pytest.ini                   # Pytest test markers & paths
└── requirements.txt             # Production dependencies
```

---

## 🤝 Contributing
Contributions, issue reports, and feature requests are welcome! See [`CONTRIBUTING.md`](./CONTRIBUTING.md) for local development, testing guidelines, and submission steps.

---

## 📄 License
This project is licensed under the [MIT License](./LICENSE).
