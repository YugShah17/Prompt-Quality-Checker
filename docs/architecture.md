# System Architecture

## Intelligent Prompt Quality Assessment Framework for Enterprise Generative AI Systems

### 1. Conceptual High-Level Architecture
```text
                  USER
                    │
                    ▼
              ┌───────────┐
              │ Frontend  │ (HTML/CSS/JS & React SPA Dashboard)
              └─────┬─────┘
                    │ REST API (JSON)
                    ▼
              ┌───────────┐
              │  Backend  │ (Python Flask / REST Routes)
              └─────┬─────┘
                    │
             ┌──────┴──────┐
             ▼             ▼
      ┌────────────┐  ┌──────────────┐
      │ ML Model   │  │ Prompt       │
      │ Assessment │  │ Improvement  │
      └─────┬──────┘  └──────┬───────┘
            │                 │
            └────────┬────────┘
                     ▼
              ┌─────────────┐
              │   SQLite    │ (database/prompt_quality.db)
              │   Database  │
              └─────────────┘
```

### 2. Core Modules & Responsibilities
- **Frontend (`frontend/` & `src/`)**: Clean, responsive user interface allowing prompt input, domain selection, visualization of the 7 quality dimensions, improvement suggestions, and historical logs.
- **Backend (`backend/` & `app.py`)**: Lightweight Flask API registering `/api/assess`, `/api/history`, and `/api/health`.
- **Machine Learning (`ml/` & `model/`)**:
  - `ml/preprocessing.py`: Cleans raw prompt inputs and normalizes formatting.
  - `ml/feature_extraction.py`: Converts text into a 22-dimensional numerical vector.
  - `ml/prediction.py`: Evaluates the prompt across 7 quality dimensions and predicts overall score (0–100) and category.
  - `ml/evaluation.py`: Evaluates MAE, RMSE, and R² metrics.
- **Prompt Improvement (`ml/prompt_improvement.py`)**: Local deterministic humanized prompt reconstruction without external LLM APIs.
- **Database (`database/` & `backend/database.py`)**: Local SQLite persistence for assessment history.
