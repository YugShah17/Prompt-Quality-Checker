# Project Workflow & Machine Learning Lifecycle

## Intelligent Prompt Quality Assessment Framework

### 1. End-to-End Pipeline
```text
User Prompt
   ↓
Frontend (templates/index.html + static/script.js)
   ↓
Flask REST Controller (app.py: /analyze)
   ↓
Text Preprocessing (utils/preprocessing.py)
   ↓
Feature Extraction (utils/feature_extraction.py - 12 Explainable Features)
   ↓
Scikit-Learn Random Forest Regressor Inference (model/predictor.py)
   ↓
Quality Assessment (Clarity, Specificity, Completeness, Relevance, Overall Score)
   ↓
Rule-Based Suggestions & Deterministic Improved Prompt (utils/suggestions.py)
   ↓
SQLite Persistence (database/prompt_quality.db)
   ↓
Frontend Result Visualization & Historical Table Log
```

### 2. Feature Extraction Details
The ML model extracts 12 explainable, interpretable features:
- **`word_count`**: Total word count.
- **`char_count`**: Total character count.
- **`sentence_count`**: Number of parsed sentences.
- **`question_marks`**: Count of question marks.
- **`instruction_words`**: Frequency of imperative/action verbs (`explain`, `analyze`, `write`, `calculate`, etc.).
- **`specificity_keywords`**: Numbers, percentages, and domain-specific terminology.
- **`context_indicators`**: Background and scenario indicators (`background`, `scenario`, `organization`, etc.).
- **`audience_indicators`**: Target persona mentions (`students`, `developers`, `executives`, etc.).
- **`format_indicators`**: Explicit format structure (`table`, `json`, `bullet points`, etc.).
- **`constraint_indicators`**: Constraint keywords (`under 300 words`, `limit to`, `formal tone`, etc.).
- **`example_indicators`**: Example references (`e.g.`, `such as`, `examples`, etc.).
- **`avg_word_length`**: Ratio of total characters in words to total word count.

### 3. Model Training Lifecycle (`model/train_model.py`)
1. **Dataset:** `data/prompts.csv` containing 150 prompt records across Poor, Needs Improvement, Good, and Excellent quality.
2. **Train/Test Split:** 80% training set (120 samples) and 20% held-out test set (30 samples).
3. **Algorithm:** Scikit-learn `RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)`.
4. **Evaluation:** Tested on held-out test data, achieving:
   - **MAE:** 1.785
   - **R² Score:** 0.9906
5. **Model Serialization:** Exported using `joblib` to `model/prompt_quality_model.pkl`.
