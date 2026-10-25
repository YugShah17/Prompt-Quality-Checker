# System Architecture & Technical Workflow

**Project Title:** Intelligent Prompt Quality Assessment Framework for Enterprise Generative AI Systems  
**Academic Level:** College-Level Machine Learning Project  
**Core Technologies:** Python, Flask, Scikit-learn (Random Forest Regressor), SQLite, HTML/CSS/JavaScript  

---

## 1. High-Level Architecture Diagram

```text
Choose Domain (HR / Finance / Marketing / General)
     ↓
User Prompt
     ↓
Frontend (HTML5 / CSS3 / JavaScript)
     ↓
Flask Backend (app.py)
     ↓
Text Preprocessing (utils/preprocessing.py)
     ↓
Feature Extraction (utils/feature_extraction.py - 13 features)
     ↓
Machine Learning Model (Random Forest Regressor)
     ↓
Domain-Aware Quality Assessment (Clarity, Specificity, Completeness, Relevance, Overall Score)
     ↓
Domain-Specific Suggestions & Improved Prompt (utils/suggestions.py)
     ↓
SQLite Database (database/prompt_quality.db)
     ↓
Result Display / History View (Displaying Selected Domain)
```

---

## 2. Component Responsibilities

| Component | Files | Primary Responsibility |
| :--- | :--- | :--- |
| **User Interface** | `templates/index.html`, `templates/result.html`, `templates/history.html`, `static/style.css`, `static/script.js` | Provides interactive domain selector tabs (HR, Finance, Marketing, General), prompt input, real-time counters, dimension score bars, domain badges, rule-based suggestion lists, improved prompt copy tool, and historical evaluation logging. |
| **Flask REST Backend** | `app.py` | Receives client requests (`/analyze`, `/import`, `/export/csv`, `/export/json`, `/history`, `/health`), validates domain and input, orchestrates ML inference, handles SQLite persistence, and serves Jinja2 views or JSON payloads. |
| **Text Preprocessing** | `utils/preprocessing.py` | Cleans irregular whitespace, handles text normalization, extracts word tokens, and splits sentences deterministically. |
| **Feature Extraction** | `utils/feature_extraction.py` | Computes 13 explainable linguistic, structural, and domain-relevance features (word count, action verbs, specificity terms, audience indicators, context markers, formatting directives, constraint tokens, domain keywords). |
| **Machine Learning Model** | `model/train_model.py`, `model/predictor.py`, `model/prompt_quality_model.pkl` | Trains and executes a Multi-Output **Random Forest Regressor** to predict continuous quality dimensions and overall quality score from tabular feature vectors. |
| **Suggestion Engine** | `utils/suggestions.py` | Detects deficiencies (e.g. missing action verb, missing audience, missing domain parameters, no output format) and provides actionable bullet points without external LLM calls. |
| **Prompt Synthesizer** | `utils/suggestions.py` | Constructs a high-clarity improved prompt using deterministic enterprise templates, injecting role persona, domain parameters, audience, and constraints. |
| **Database Storage** | `database/db.py`, `database/prompt_quality.db` | Persists evaluated prompts, selected enterprise domain, 4 dimension scores, overall score, quality category, suggestions, and improved prompts into a lightweight SQLite `assessments` table. |

---

## 3. Data Flow

1. **Domain Selection & User Submission:** The user selects an enterprise domain (**HR**, **Finance**, **Marketing**, or **General**) and enters a prompt.
2. **HTTP Request:** The frontend validates input length and transmits a `POST` request to `/analyze` with payload:
   ```json
   {
     "prompt": "Draft 5 behavioral interview questions using the STAR framework for hiring a senior engineer.",
     "domain": "HR"
   }
   ```
3. **Validation & Preprocessing:** Flask validates domain and input length. `utils.preprocessing` cleans whitespace and tokenizes words and sentences.
4. **Feature Vectorization:** `utils.feature_extraction` scans the tokens against explainable linguistic dictionaries to produce a 12-dimensional numerical vector:
   `[word_count, char_count, sentence_count, question_marks, instruction_words, specificity_keywords, context_indicators, audience_indicators, format_indicators, constraint_indicators, example_indicators, avg_word_length]`
5. **Random Forest Inference:** The 12-dimensional feature vector is passed to the trained `RandomForestRegressor`. The model predicts continuous scores for:
   - Clarity (0–100)
   - Specificity (0–100)
   - Completeness (0–100)
   - Relevance (0–100)
6. **Overall Score Calculation:**
   $$\text{Overall Score} = 0.30 \times \text{Clarity} + 0.30 \times \text{Specificity} + 0.20 \times \text{Completeness} + 0.20 \times \text{Relevance}$$
   The score is clamped between 0 and 100.
7. **Categorization:**
   - `0–39` → **Poor**
   - `40–59` → **Needs Improvement**
   - `60–79` → **Good**
   - `80–100` → **Excellent**
8. **Suggestion Generation:** Heuristic rules inspect missing features (e.g. lack of target audience or context) and generate clear recommendations.
9. **Prompt Improvement:** Deterministic template logic synthesizes a structured, higher-quality version of the prompt.
10. **Database Persistence:** The prompt, dimension scores, overall score, category, suggestions list, and improved prompt are saved to `database/prompt_quality.db`.
11. **Client Response:** Flask returns HTTP 200 with JSON data and the frontend renders the interactive results.

---

## 4. Machine Learning Workflow

```text
data/prompts.csv (150 Diverse Prompt Records)
     ↓
Feature Extraction (utils/feature_extraction.py)
     ↓
Train/Test Split (80% Train, 20% Test)
     ↓
RandomForestRegressor (n_estimators=100, max_depth=10, random_state=42)
     ↓
Evaluation on Test Set (MAE = 1.785, R² = 0.9906)
     ↓
Model Serialization (model/prompt_quality_model.pkl via joblib)
     ↓
Inference Predictor (model/predictor.py)
```

### Explainable Features Used
1. `word_count`: Total number of tokens in the prompt.
2. `char_count`: Total character length.
3. `sentence_count`: Sentence division count.
4. `question_marks`: Count of `?` characters.
5. `instruction_words`: Frequency of action verbs (`explain`, `analyze`, `compare`, `summarize`, etc.).
6. `specificity_keywords`: Numerical digits, percentages, and domain terminology.
7. `context_indicators`: Background and scenario tokens (`background`, `enterprise`, `scenario`, etc.).
8. `audience_indicators`: Target persona mentions (`students`, `developers`, `executives`, etc.).
9. `format_indicators`: Explicit output structure (`table`, `json`, `bullet points`, etc.).
10. `constraint_indicators`: Boundaries (`under 300 words`, `limit to`, `formal tone`, etc.).
11. `example_indicators`: Reference tokens (`e.g.`, `such as`, `examples`, etc.).
12. `avg_word_length`: Ratio of total characters in words to total word count.

---

## 5. Database Schema (SQLite)

Database file: `database/prompt_quality.db`  
Table name: `assessments`

```sql
CREATE TABLE IF NOT EXISTS assessments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt TEXT NOT NULL,
    clarity REAL NOT NULL,
    specificity REAL NOT NULL,
    completeness REAL NOT NULL,
    relevance REAL NOT NULL,
    overall_score REAL NOT NULL,
    category TEXT NOT NULL,
    suggestions TEXT NOT NULL,
    improved_prompt TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## 6. User Workflow

1. **Access Application:** User opens `http://127.0.0.1:5000` in any web browser.
2. **Input Prompt:** User enters a custom prompt or clicks one of the quick test chips (e.g. *⭐ Excellent Prompt Example*, *⚠️ Poor Prompt Example*).
3. **Assessment:** User clicks **Analyze Prompt**.
4. **Inspect Results:**
   - Overall Quality Score (0–100 gauge)
   - Quality Category badge (*Poor*, *Needs Improvement*, *Good*, or *Excellent*)
   - Four dimension progress bars (*Clarity*, *Specificity*, *Completeness*, *Relevance*)
   - Bulleted list of practical suggestions
   - Copyable Improved Prompt
5. **Re-Test Improved Prompt:** User clicks "Test This Improved Prompt" to immediately verify score enhancement.
6. **Review History:** User navigates to `/history` to inspect all previous prompt evaluations logged in SQLite.
