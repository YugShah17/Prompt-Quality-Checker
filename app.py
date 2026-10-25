"""
Intelligent Prompt Quality Assessment Framework - Flask Application
Clean, simple, local Machine Learning application for college project.

Endpoints:
  GET    /             - Page 1: Assessment Interface
  GET    /result       - Page 2: Assessment Result View
  GET    /history      - Page 3: History Table (or JSON API)
  POST   /analyze      - Core ML Inference & SQLite Persistence Endpoint
  DELETE /history      - Clear Evaluation History
  GET    /health       - System Health & Model Status
"""
import os
import sys
from flask import Flask, request, jsonify, render_template, redirect, url_for

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.predictor import PromptQualityPredictor
from database.db import (
    init_db,
    save_assessment,
    get_all_assessments,
    get_assessment_by_id,
    clear_all_assessments
)

# Initialize Flask app
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["SECRET_KEY"] = "college-ml-project-secret"

# Initialize SQLite database schema
init_db()

# Load trained Machine Learning model
predictor = PromptQualityPredictor()

# Keep track of latest assessment in memory for optional /result page view
latest_result = None

@app.route("/", methods=["GET"])
def index():
    """Page 1 — Assessment Form Page."""
    return render_template("index.html")

@app.route("/result", methods=["GET"])
def result_page():
    """Page 2 — Result Display Page."""
    global latest_result
    record_id = request.args.get("id")
    res = None
    if record_id:
        try:
            res = get_assessment_by_id(int(record_id))
        except Exception:
            res = None
    if not res:
        res = latest_result

    if not res:
        return redirect(url_for("index"))
    return render_template("result.html", result=res)

@app.route("/history", methods=["GET", "DELETE"])
@app.route("/api/history", methods=["GET", "DELETE"])
def history_endpoint():
    """
    Page 3 — History View or JSON API.
    GET: Returns HTML page or JSON array.
    DELETE: Clears all records.
    """
    if request.method == "DELETE":
        clear_all_assessments()
        return jsonify({"success": True, "message": "Assessment history cleared successfully."}), 200

    limit = request.args.get("limit", default=50, type=int)
    history_items = get_all_assessments(limit=limit)

    # Return JSON if requested by API client or fetch
    wants_json = (
        request.headers.get("Accept", "").find("application/json") != -1 or
        request.args.get("format") == "json" or
        request.path.startswith("/api/")
    )
    if wants_json:
        return jsonify(history_items), 200

    return render_template("history.html", history=history_items)

@app.route("/analyze", methods=["POST"])
@app.route("/api/analyze", methods=["POST"])
@app.route("/api/assess", methods=["POST"])
def analyze_endpoint():
    """
    Core ML Assessment Endpoint.
    Workflow:
      1. Receive prompt (JSON or Form)
      2. Validate prompt
      3. Call ML Predictor (feature extraction + Random Forest inference)
      4. Calculate quality dimensions & overall score
      5. Generate rule-based suggestions & deterministic improved prompt
      6. Save result to SQLite
      7. Return JSON response
    """
    global latest_result

    # 1. Receive prompt and domain
    valid_domains = ["HR", "Finance", "Marketing", "General"]
    domain = "General"
    if request.is_json:
        data = request.get_json(silent=True) or {}
        prompt = data.get("prompt", "")
        domain = data.get("domain", "General")
    else:
        prompt = request.form.get("prompt", "")
        domain = request.form.get("domain", "General")

    if domain not in valid_domains:
        domain = "General"

    # 2. Validate prompt
    clean_prompt = prompt.strip() if prompt else ""
    if not clean_prompt:
        return jsonify({"error": "Prompt cannot be empty. Please enter a valid prompt."}), 400

    if len(clean_prompt) < 3:
        return jsonify({"error": "Prompt is too short. Please provide a meaningful prompt."}), 400

    # 3. Model prediction (Features -> Random Forest -> Dimensions & Overall Score)
    res = predictor.predict(clean_prompt, domain=domain)

    # 4. Persist to SQLite database
    record_id = None
    try:
        record_id = save_assessment(
            prompt=clean_prompt,
            clarity=res["clarity"],
            specificity=res["specificity"],
            completeness=res["completeness"],
            relevance=res["relevance"],
            overall_score=res["overall_score"],
            category=res["category"],
            suggestions=res["suggestions"],
            improved_prompt=res["improved_prompt"],
            domain=res["domain"]
        )
    except Exception as e:
        print(f"Database error while saving assessment: {e}")

    res["id"] = record_id
    latest_result = res

    # If submitted via standard HTML form, redirect to result page
    if not request.is_json and request.headers.get("Accept", "").find("application/json") == -1:
        return redirect(url_for("result_page", id=record_id))

    return jsonify(res), 200

@app.route("/import", methods=["POST"])
@app.route("/api/import", methods=["POST"])
def import_file_endpoint():
    """
    Import File Endpoint.
    Accepts CSV, TXT, or JSON file containing prompts.
    Evaluates each prompt through existing ML pipeline and stores in SQLite.
    """
    import io
    import csv
    import json

    global latest_result

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Please select a CSV, TXT, or JSON file."}), 400

    uploaded_file = request.files["file"]
    filename = uploaded_file.filename.lower()
    content_bytes = uploaded_file.read()

    valid_domains = ["HR", "Finance", "Marketing", "General"]
    domain = request.form.get("domain") or request.args.get("domain") or "General"
    if domain not in valid_domains:
        domain = "General"

    if not content_bytes:
        return jsonify({"error": "Uploaded file is empty."}), 400

    prompts = []
    text = content_bytes.decode("utf-8", errors="ignore").strip()

    if filename.endswith(".csv"):
        reader = csv.reader(io.StringIO(text))
        prompt_idx = 0
        for i, row in enumerate(reader):
            if not row or not any(row):
                continue
            if i == 0:
                header = [col.strip().lower() for col in row]
                if "prompt" in header:
                    prompt_idx = header.index("prompt")
                    continue
                else:
                    prompts.append(row[0].strip())
            else:
                idx = prompt_idx if prompt_idx < len(row) else 0
                val = row[idx].strip()
                if val:
                    prompts.append(val)
    elif filename.endswith(".json"):
        try:
            data = json.loads(text)
        except Exception as e:
            return jsonify({"error": f"Invalid JSON format: {str(e)}"}), 400

        if isinstance(data, list):
            for item in data:
                if isinstance(item, str) and item.strip():
                    prompts.append(item.strip())
                elif isinstance(item, dict):
                    p = item.get("prompt") or item.get("text") or ""
                    if p.strip():
                        prompts.append(p.strip())
        elif isinstance(data, dict):
            items = data.get("prompts") or data.get("data") or []
            for item in items:
                if isinstance(item, str) and item.strip():
                    prompts.append(item.strip())
                elif isinstance(item, dict):
                    p = item.get("prompt") or item.get("text") or ""
                    if p.strip():
                        prompts.append(p.strip())
    else:  # Treat as TXT
        for line in text.splitlines():
            line = line.strip()
            if line:
                prompts.append(line)

    if not prompts:
        return jsonify({"error": "No prompts found in the imported file."}), 400

    results = []
    for p in prompts:
        if len(p.strip()) < 3:
            continue
        res = predictor.predict(p.strip(), domain=domain)
        record_id = None
        try:
            record_id = save_assessment(
                prompt=p.strip(),
                clarity=res["clarity"],
                specificity=res["specificity"],
                completeness=res["completeness"],
                relevance=res["relevance"],
                overall_score=res["overall_score"],
                category=res["category"],
                suggestions=res["suggestions"],
                improved_prompt=res["improved_prompt"],
                domain=res["domain"]
            )
        except Exception as e:
            print(f"Error saving imported assessment: {e}")

        res["id"] = record_id
        results.append(res)

    if not results:
        return jsonify({"error": "All prompts in file were too short or invalid."}), 400

    latest_result = results[-1]
    return jsonify({
        "success": True,
        "count": len(results),
        "results": results
    }), 200

@app.route("/export/csv", methods=["GET", "POST"])
@app.route("/api/export/csv", methods=["GET", "POST"])
def export_csv_endpoint():
    """
    Export Assessment Results as CSV.
    Columns: domain, prompt, clarity, specificity, completeness, relevance,
             overall_score, category, suggestions, improved_prompt
    """
    import io
    import csv
    import json
    from flask import Response

    assessments = get_all_assessments(limit=1000)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "domain", "prompt", "clarity", "specificity", "completeness", "relevance",
        "overall_score", "category", "suggestions", "improved_prompt"
    ])

    for a in assessments:
        sugs = a.get("suggestions", [])
        sug_str = "; ".join(sugs) if isinstance(sugs, list) else str(sugs)
        writer.writerow([
            a.get("domain", "General"),
            a.get("prompt", ""),
            a.get("clarity", 0.0),
            a.get("specificity", 0.0),
            a.get("completeness", 0.0),
            a.get("relevance", 0.0),
            int(round(float(a.get("overall_score", 0)))),
            a.get("category", ""),
            sug_str,
            a.get("improved_prompt", "")
        ])

    csv_data = output.getvalue()
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=prompt_assessment_results.csv"}
    )

@app.route("/export/json", methods=["GET", "POST"])
@app.route("/api/export/json", methods=["GET", "POST"])
def export_json_endpoint():
    """
    Export Assessment Results as JSON.
    """
    import json
    from flask import Response

    assessments = get_all_assessments(limit=1000)
    export_data = []

    for a in assessments:
        export_data.append({
            "domain": a.get("domain", "General"),
            "prompt": a.get("prompt", ""),
            "clarity": a.get("clarity", 0.0),
            "specificity": a.get("specificity", 0.0),
            "completeness": a.get("completeness", 0.0),
            "relevance": a.get("relevance", 0.0),
            "overall_score": int(round(float(a.get("overall_score", 0)))),
            "category": a.get("category", ""),
            "suggestions": a.get("suggestions", []),
            "improved_prompt": a.get("improved_prompt", "")
        })

    json_str = json.dumps(export_data, indent=2)
    return Response(
        json_str,
        mimetype="application/json",
        headers={"Content-Disposition": "attachment; filename=prompt_assessment_results.json"}
    )

@app.route("/health", methods=["GET"])
@app.route("/api/health", methods=["GET"])
def health():
    """System Health and ML Model Status Check."""
    is_model_loaded = predictor.model is not None
    return jsonify({
        "status": "healthy",
        "project": "Intelligent Prompt Quality Assessment Framework",
        "domains": ["HR", "Finance", "Marketing", "General"],
        "model_type": "Random Forest Regressor (Scikit-learn)",
        "model_loaded": is_model_loaded,
        "database": "SQLite (database/prompt_quality.db)",
        "features": 13,
        "external_ai_dependencies": "None (100% Local Inference)"
    }), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 65)
    print(" Intelligent Prompt Quality Assessment Framework (Flask)")
    print(f" Server running at: http://127.0.0.1:{port}")
    print("=" * 65)
    app.run(host="127.0.0.1", port=port, debug=False)
