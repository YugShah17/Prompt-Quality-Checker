"""
SQLite Database Module for Prompt Quality Assessment records.
Database file: database/prompt_quality.db
Table: assessments
"""
import os
import sqlite3
import json

DB_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DB_DIR, "prompt_quality.db")

def get_connection():
    """Returns a connection to the SQLite database."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the SQLite database table according to specification."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS assessments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        domain TEXT DEFAULT 'General',
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
    )
    """)
    # Schema migration: Add domain column if upgrading from earlier table
    cursor.execute("PRAGMA table_info(assessments)")
    cols = [col[1] for col in cursor.fetchall()]
    if "domain" not in cols:
        cursor.execute("ALTER TABLE assessments ADD COLUMN domain TEXT DEFAULT 'General'")

    conn.commit()
    conn.close()

def save_assessment(prompt, clarity, specificity, completeness, relevance,
                    overall_score, category, suggestions, improved_prompt, domain="General"):
    """
    Saves a newly evaluated assessment record to the database.
    Returns the newly inserted record ID.
    """
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    sug_str = json.dumps(suggestions) if isinstance(suggestions, list) else str(suggestions)
    valid_domains = ["HR", "Finance", "Marketing", "General"]
    clean_domain = domain if domain in valid_domains else "General"

    cursor.execute("""
    INSERT INTO assessments (
        domain, prompt, clarity, specificity, completeness, relevance,
        overall_score, category, suggestions, improved_prompt, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    """, (
        clean_domain, str(prompt), float(clarity), float(specificity),
        float(completeness), float(relevance), float(overall_score),
        str(category), sug_str, str(improved_prompt)
    ))
    record_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return record_id

def get_all_assessments(limit=50):
    """
    Retrieves previous assessments sorted by created_at descending.
    """
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT id, domain, prompt, clarity, specificity, completeness, relevance,
           overall_score, category, suggestions, improved_prompt, created_at
    FROM assessments
    ORDER BY created_at DESC
    LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    results = []
    for r in rows:
        try:
            sugs = json.loads(r["suggestions"])
        except Exception:
            sugs = [r["suggestions"]] if r["suggestions"] else []
        results.append({
            "id": r["id"],
            "domain": r["domain"] if "domain" in r.keys() and r["domain"] else "General",
            "prompt": r["prompt"],
            "clarity": r["clarity"],
            "specificity": r["specificity"],
            "completeness": r["completeness"],
            "relevance": r["relevance"],
            "overall_score": int(round(r["overall_score"])),
            "category": r["category"],
            "suggestions": sugs,
            "improved_prompt": r["improved_prompt"],
            "created_at": str(r["created_at"])
        })
    conn.close()
    return results

def get_assessment_by_id(record_id):
    """Retrieves a single assessment by ID."""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT id, domain, prompt, clarity, specificity, completeness, relevance,
           overall_score, category, suggestions, improved_prompt, created_at
    FROM assessments
    WHERE id = ?
    """, (record_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    try:
        sugs = json.loads(row["suggestions"])
    except Exception:
        sugs = [row["suggestions"]] if row["suggestions"] else []
    return {
        "id": row["id"],
        "domain": row["domain"] if "domain" in row.keys() and row["domain"] else "General",
        "prompt": row["prompt"],
        "clarity": row["clarity"],
        "specificity": row["specificity"],
        "completeness": row["completeness"],
        "relevance": row["relevance"],
        "overall_score": int(round(row["overall_score"])),
        "category": row["category"],
        "suggestions": sugs,
        "improved_prompt": row["improved_prompt"],
        "created_at": str(row["created_at"])
    }


def delete_assessment(record_id):
    """Deletes an assessment by ID."""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM assessments WHERE id = ?", (record_id,))
    conn.commit()
    conn.close()

def clear_all_assessments():
    """Deletes all records from assessments table."""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM assessments")
    conn.commit()
    conn.close()
