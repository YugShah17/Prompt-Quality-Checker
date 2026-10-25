# Intelligent Prompt Quality Assessment Framework for Enterprise Generative AI Systems

> **College Machine Learning Project**  
> A lightweight, explainable, and 100% offline Machine Learning application built using **Python, Flask, Scikit-learn (Random Forest Regressor), and SQLite**.

---

## 1. Project Title
**Intelligent Prompt Quality Assessment Framework for Enterprise Generative AI Systems**

---

## 2. Problem Statement
In generative AI workflows, user prompts are often vague, underspecified, or lack critical contextual parameters such as persona, target audience, format directives, and constraints. Poorly structured prompts lead to suboptimal model responses, hallucinations, unnecessary latency, and wasted computational resources. Most existing prompt optimizers rely on expensive, non-deterministic cloud LLM APIs (e.g., OpenAI or Gemini), creating vendor lock-in, data privacy risks, and lack of mathematical explainability.

---

## 3. Objective
To design and deploy a lightweight, interpretable, and self-contained Machine Learning system that:
- Evaluates raw text prompts across four fundamental quality dimensions.
- Predicts an overall quality score between **0 and 100** using a trained **Random Forest Regressor**.
- Assigns a clear quality category (*Poor*, *Needs Improvement*, *Good*, *Excellent*).
- Delivers explainable, rule-based improvement suggestions.
- Generates deterministic improved prompts without external LLM dependencies.
- Persists all assessment records in a local SQLite database.

---

## 4. Proposed Solution
An end-to-end framework combining deterministic Natural Language Processing (NLP) feature engineering with Scikit-learn's `RandomForestRegressor`. The system extracts 12 explainable structural and linguistic features from prompt text, infers quality dimensions, calculates an overall score via a convex weighted formulation, suggests missing elements, and logs results into SQLite.

---

## 5. Key Features
- **100% Local Inference:** No OpenAI, Gemini, Claude, API keys, or internet connection required.
- **Enterprise Domain Selection:** Dedicated support for **HR**, **Finance**, **Marketing**, and **General** enterprise domains.
- **Explainable Machine Learning:** Uses Scikit-learn's Random Forest algorithm with 13 interpretable feature vectors (including domain relevance).
- **Multi-Dimensional Quality Grading:** Clarity (30%), Specificity (30%), Completeness (20%), Relevance (20%).
- **Domain-Specific Suggestion Engine:** Automatically identifies missing domain context, audience, format, or constraints.
- **Domain-Tailored Improved Prompt:** Instantly provides a structured alternative with an appropriate enterprise persona.
- **Import and Export File:** Batch-import prompts from CSV, TXT, or JSON files and export assessment results to CSV or JSON with domain tags.
- **SQLite Historical Persistence:** Automatically logs assessments in `database/prompt_quality.db`.
- **Clean Three-Page Interface:** Assessment (`/`), Result (`/result`), and History (`/history`).
- **RESTful API:** Supports `/analyze`, `/import`, `/export/csv`, `/export/json`, `/history`, and `/health` endpoints.

---

## 6. Technology Stack
- **Frontend:** HTML5, Vanilla CSS3, Vanilla JavaScript (no heavy frontend frameworks)
- **Backend:** Python 3.10+, Flask
- **Machine Learning:** Scikit-learn (`RandomForestRegressor`), Joblib
- **Data Processing:** Pandas, NumPy
- **Database:** SQLite3
- **Testing:** Python `unittest`

---

## 7. Architecture Overview
```text
Choose Domain (HR / Finance / Marketing / General)
     ↓
User Prompt
     ↓
Frontend (HTML / CSS / JavaScript)
     ↓
Flask Backend (app.py)
     ↓
Text Preprocessing (utils/preprocessing.py)
     ↓
Feature Extraction (utils/feature_extraction.py - 13 features)
     ↓
Machine Learning Model (Random Forest Regressor)
     ↓
Domain-Aware Quality Assessment (Clarity, Specificity, Completeness, Relevance)
     ↓
Domain-Specific Suggestions & Improved Prompt (utils/suggestions.py)
     ↓
SQLite Database (database/prompt_quality.db)
     ↓
Result / History Views (Showing Selected Domain)
```

---

## 8. Workflow
**Choose Domain → Enter Prompt → Assess Prompt → Show Quality Results**

1. User selects an enterprise domain (**HR**, **Finance**, **Marketing**, or **General**).
2. User enters a domain-related prompt in the interface.
3. System extracts 13 linguistic and domain-alignment features.
4. Trained Random Forest Regressor evaluates quality dimensions.
5. Overall score and quality category are determined.
6. Domain-tailored suggestions and improved prompts are generated.
7. Result is displayed with the selected domain badge and saved in SQLite.

---

## 9. Machine Learning Model
- **Algorithm:** `RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)`
- **Input:** 13-dimensional tabular feature vector (linguistic + domain relevance).
- **Output:** Continuous predictions for `[clarity, specificity, completeness, relevance, quality_score]`.
- **Artifact:** Serialized with `joblib` into `model/prompt_quality_model.pkl`.

---

## 10. Dataset
- **File:** `data/prompts.csv`
- **Rows:** 140 enterprise prompt samples balanced across the 4 domains:
  - **HR (35 samples):** Recruitment, employee management, job descriptions, performance evaluation, training.
  - **Finance (35 samples):** Financial reports, budgeting, expense analysis, financial data, business finance.
  - **Marketing (35 samples):** Marketing campaigns, customer communication, advertising, content creation, market analysis.
  - **General (35 samples):** General enterprise workflows, procedures, and cross-functional operations.
- **Columns:** `domain, prompt, clarity, specificity, completeness, relevance, quality_score`
  - `prompt`: Raw text string.
  - `clarity`: Clarity ground-truth score (0–100).
  - `specificity`: Specificity ground-truth score (0–100).
  - `completeness`: Completeness ground-truth score (0–100).
  - `relevance`: Relevance ground-truth score (0–100).
  - `quality_score`: Overall composite target score (0–100).

---

## 11. Feature Extraction
The framework extracts 12 explainable features:
1. `word_count`: Total word count.
2. `char_count`: Total character count.
3. `sentence_count`: Number of parsed sentences.
4. `question_marks`: Count of question marks.
5. `instruction_words`: Frequency of action verbs (`explain`, `analyze`, `write`, `calculate`, etc.).
6. `specificity_keywords`: Numerical digits, percentages, technical keywords.
7. `context_indicators`: Scenario tokens (`background`, `scenario`, `organization`, etc.).
8. `audience_indicators`: Target persona mentions (`students`, `developers`, `executives`, etc.).
9. `format_indicators`: Explicit output structure (`table`, `json`, `bullet points`, etc.).
10. `constraint_indicators`: Boundary words (`under 300 words`, `limit to`, `formal tone`, etc.).
11. `example_indicators`: Example references (`e.g.`, `such as`, `examples`, etc.).
12. `avg_word_length`: Average characters per word.

---

## 12. Database Schema
Database file: `database/prompt_quality.db`  
Table: `assessments`

| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | INTEGER | Auto-increment primary key |
| `prompt` | TEXT | Evaluated prompt text |
| `clarity` | REAL | Clarity dimension score (0–100) |
| `specificity` | REAL | Specificity dimension score (0–100) |
| `completeness` | REAL | Completeness dimension score (0–100) |
| `relevance` | REAL | Relevance dimension score (0–100) |
| `overall_score` | REAL | Weighted overall score (0–100) |
| `category` | TEXT | Quality tier (Poor / Needs Improvement / Good / Excellent) |
| `suggestions` | TEXT | JSON list of rule-based suggestions |
| `improved_prompt` | TEXT | Synthesized improved prompt |
| `created_at` | TIMESTAMP | Record insertion timestamp |

---

## 13. File Structure
```text
ML Project/
│
├── app.py                      # Flask backend and route controllers
├── requirements.txt            # Python dependencies
├── README.md                   # Complete project documentation
├── architecture.md             # System architecture & workflow guide
├── train_model.py              # Root convenience training runner
│
├── database/
│   ├── db.py                   # SQLite schema initialization and CRUD
│   └── prompt_quality.db       # SQLite database file
│
├── data/
│   └── prompts.csv             # 150 prompt training dataset
│
├── model/
│   ├── train_model.py          # Random Forest training & evaluation script
│   ├── predictor.py            # Model inference engine & scoring logic
│   └── prompt_quality_model.pkl # Trained Random Forest artifact
│
├── utils/
│   ├── preprocessing.py        # Text cleaning and tokenization
│   ├── feature_extraction.py   # 12 explainable NLP features
│   └── suggestions.py          # Suggestion engine & improved prompt builder
│
├── templates/
│   ├── index.html              # Page 1: Assessment Interface
│   ├── result.html             # Page 2: Result Display
│   └── history.html            # Page 3: History Table
│
├── static/
│   ├── style.css               # Clean responsive styling
│   └── script.js               # Frontend interaction and AJAX handler
│
└── tests/
    ├── __init__.py
    └── test_model_pipeline.py  # Automated unit test suite
```

---

## 14. Installation

1. Open terminal in the project directory:
   ```bash
   cd "c:\Users\Pcm\OneDrive\Desktop\ML Project"
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 15. Running Instructions

### Step 1: Train the Machine Learning Model
```bash
python model/train_model.py
```
*Loads `data/prompts.csv`, extracts features, trains the Random Forest Regressor, evaluates MAE and R², and outputs `model/prompt_quality_model.pkl`.*

### Step 2: Start the Flask Application
```bash
python app.py
```

### Step 3: Access the Application
Open your browser and navigate to:
```text
http://127.0.0.1:5000
```

### Step 4: Run Automated Tests
```bash
python -m unittest discover tests
```

---

## 16. API Endpoints

### 1. `POST /analyze` (or `/api/analyze`)
Evaluates a prompt, extracts features, performs Random Forest inference, stores the result in SQLite, and returns JSON.

**Request:**
```bash
curl -X POST http://127.0.0.1:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain artificial intelligence to first-year computer science students in simple language with three real-world examples."}'
```

**Response (HTTP 200):**
```json
{
  "id": 1,
  "prompt": "Explain artificial intelligence to first-year computer science students in simple language with three real-world examples.",
  "clarity": 81.8,
  "specificity": 78.6,
  "completeness": 75.1,
  "relevance": 84.1,
  "overall_score": 80.0,
  "category": "Excellent",
  "suggestions": [
    "Provide more context (e.g., background scenario, domain setting, or project scope).",
    "Specify the expected output format (e.g., bullet points, markdown table, step-by-step list)."
  ],
  "improved_prompt": "Explain artificial intelligence to first-year computer science students in simple language with three real-world examples."
}
```

### 2. `POST /import` (or `/api/import`)
Imports a CSV, TXT, or JSON file containing prompts, evaluates each prompt via the existing ML pipeline, stores the results in SQLite, and returns the evaluation list.

### 3. `GET /export/csv` (or `/api/export/csv`)
Exports all evaluated prompt assessment results as a downloadable CSV file (`prompt_assessment_results.csv`).

### 4. `GET /export/json` (or `/api/export/json`)
Exports all evaluated prompt assessment results as a formatted JSON file (`prompt_assessment_results.json`).

### 5. `GET /history` (or `/api/history`)
Returns all historical assessments from SQLite.

### 6. `DELETE /history` (or `/api/history`)
Clears all assessment records from SQLite.

### 7. `GET /health` (or `/api/health`)
Returns backend operational status and ML model health.


---

## 17. Sample Input & Output

### Example A: Comprehensive (High Quality) Prompt
- **Input:**
  > `"Explain artificial intelligence to first-year computer science students in simple language with three real-world examples."`
- **Predicted Overall Score:** `80.0 / 100`
- **Category:** `Excellent`
- **Dimension Scores:**
  - Clarity: `81.8%`
  - Specificity: `78.6%`
  - Completeness: `75.1%`
  - Relevance: `84.1%`
- **Result:** Successfully categorized as Excellent and recorded in SQLite.

### Example B: Vague (Poor Quality) Prompt
- **Input:**
  > `"Tell me about AI."`
- **Predicted Overall Score:** `21.2 / 100`
- **Category:** `Poor`
- **Dimension Scores:**
  - Clarity: `23.9%`
  - Specificity: `15.0%`
  - Completeness: `18.9%`
  - Relevance: `28.7%`
- **Suggestions Generated:**
  1. Add more details to your prompt.
  2. Clearly mention what you want the system to do.
  3. Specify the target audience.
  4. Provide more context.
  5. Specify the expected output format.
  6. Add specific requirements, examples, or constraints.
- **Improved Prompt:**
  > `"Explain artificial intelligence to college students using simple language. Cover the basic concept, applications, advantages, and three real-world examples."`

---

## 18. Machine Learning Evaluation

Evaluated on a 20% held-out test split (30 test prompts from `data/prompts.csv`):

| Evaluation Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Mean Absolute Error (MAE)** | **1.785** | Predictions deviate by less than 2 points on a 0–100 scale |
| **R² Score (Coefficient of Determination)** | **0.9906** | Model accounts for >99% of prompt quality variance |
| **Model Type** | Random Forest Regressor | Ensemble of 100 decision trees |
| **Training Time** | < 0.5 seconds | Suitable for quick re-training and live demo |

---

## 19. Limitations
- Evaluates prompt structure and linguistic features; does not perform factual semantic verification.
- Feature extraction relies on English vocabulary dictionaries and regular expressions.
- The training dataset is scoped to educational, enterprise, and technical domains.

---

## 20. Future Scope
- Expand linguistic dictionary support for multi-lingual prompt evaluation.
- Incorporate domain-specific custom lexicons (e.g., healthcare HIPAA, legal drafting).
- Add automated A/B test benchmarking against target open-source models (e.g. Ollama/Llama-3).
- Provide a browser extension to grade prompts directly in AI chat interfaces.
