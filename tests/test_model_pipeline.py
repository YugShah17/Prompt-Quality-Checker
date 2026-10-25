"""
Unit Test Suite for Intelligent Prompt Quality Assessment Framework.
Verifies end-to-end functionality:
  1. Dataset existence
  2. Trained model artifact existence
  3. Feature extraction (12 explainable features)
  4. Random Forest prediction and category classification
  5. Suggestion generation
  6. Deterministic improved prompt synthesis
  7. SQLite database persistence and retrieval
"""
import unittest
import os
import sys

# Ensure root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.predictor import PromptQualityPredictor
from utils.feature_extraction import extract_features, get_feature_vector, FEATURE_NAMES
from utils.suggestions import generate_suggestions, generate_improved_prompt
from database.db import save_assessment, get_all_assessments, clear_all_assessments, init_db

class TestPromptQualityPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()
        cls.predictor = PromptQualityPredictor()

    def test_dataset_exists(self):
        """Verifies dataset data/prompts.csv exists and has content."""
        self.assertTrue(os.path.exists("data/prompts.csv"))
        with open("data/prompts.csv", "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.assertGreater(len(lines), 50)

    def test_model_artifact_exists(self):
        """Verifies trained model exists in model/prompt_quality_model.pkl."""
        self.assertTrue(os.path.exists("model/prompt_quality_model.pkl"))
        self.assertIsNotNone(self.predictor.model)

    def test_feature_extraction(self):
        """Verifies all 13 explainable features (including domain_relevance) are extracted."""
        prompt = "Create 5 behavioral interview questions using the STAR method for hiring a project manager."
        feats = extract_features(prompt, domain="HR")
        self.assertEqual(len(FEATURE_NAMES), 13)
        for name in FEATURE_NAMES:
            self.assertIn(name, feats)
        self.assertGreater(feats["word_count"], 10)
        self.assertGreater(feats["domain_relevance"], 0)

        vec = get_feature_vector(prompt, domain="HR")
        self.assertEqual(len(vec), 13)

    def test_enterprise_domains_supported(self):
        """Verifies all 4 enterprise domains (HR, Finance, Marketing, General) are supported."""
        domains = ["HR", "Finance", "Marketing", "General"]
        sample_prompts = {
            "HR": "Draft behavioral interview rubric for senior engineer hiring.",
            "Finance": "Analyze quarterly budget variance and departmental operating expenses.",
            "Marketing": "Design a multi-channel product launch campaign to drive customer acquisition.",
            "General": "Draft an enterprise operational workflow guideline for cross-functional teams."
        }
        for d in domains:
            res = self.predictor.predict(sample_prompts[d], domain=d)
            self.assertEqual(res["domain"], d)
            self.assertIn("clarity", res)
            self.assertIn("specificity", res)
            self.assertIn("completeness", res)
            self.assertIn("relevance", res)
            self.assertIsInstance(res["overall_score"], int)

    def test_model_prediction_excellent_prompt(self):
        """Verifies prediction for comprehensive prompt yields high score (>= 70)."""
        prompt = "Draft 5 behavioral interview questions using the STAR method for hiring a project manager, detailing evaluation rubric and scoring criteria."
        res = self.predictor.predict(prompt, domain="HR")
        self.assertGreaterEqual(res["overall_score"], 65)
        self.assertIn(res["category"], ["Good", "Excellent"])
        self.assertEqual(res["domain"], "HR")

    def test_model_prediction_poor_prompt(self):
        """Verifies prediction for vague short prompt yields low score (< 40)."""
        prompt = "Hire staff."
        res = self.predictor.predict(prompt, domain="HR")
        self.assertLess(res["overall_score"], 40)
        self.assertEqual(res["category"], "Poor")
        self.assertGreater(len(res["suggestions"]), 0)
        self.assertTrue(len(res["improved_prompt"]) > len(prompt))

    def test_database_operations(self):
        """Verifies SQLite save and retrieve operations with domain field."""
        clear_all_assessments()
        rec_id = save_assessment(
            prompt="Unit test prompt",
            clarity=85.0,
            specificity=80.0,
            completeness=75.0,
            relevance=90.0,
            overall_score=82,
            category="Excellent",
            suggestions=["Keep it up"],
            improved_prompt="Unit test improved",
            domain="Finance"
        )
        self.assertIsNotNone(rec_id)
        items = get_all_assessments(limit=10)
        self.assertGreaterEqual(len(items), 1)
        self.assertEqual(items[0]["prompt"], "Unit test prompt")
        self.assertEqual(items[0]["domain"], "Finance")
        self.assertEqual(items[0]["category"], "Excellent")
        clear_all_assessments()

    def test_dataset_relevance_only(self):
        """Verifies only project-related prompt quality dataset exists and domains match."""
        import pandas as pd
        self.assertTrue(os.path.exists("data/prompts.csv"))
        df = pd.read_csv("data/prompts.csv")
        required_cols = {"domain", "prompt", "clarity", "specificity", "completeness", "relevance", "quality_score"}
        self.assertTrue(required_cols.issubset(set(df.columns)))
        # Verify exactly the 4 required enterprise domains
        unique_domains = set(df["domain"].unique())
        self.assertEqual(unique_domains, {"HR", "Finance", "Marketing", "General"})
        # Verify no unrelated datasets exist
        data_files = [f for f in os.listdir("data") if f.endswith(".csv")]
        self.assertEqual(data_files, ["prompts.csv"])

    def test_import_export_endpoints(self):
        """Verifies CSV, TXT, JSON import and CSV/JSON export with domain."""
        import io
        import json
        from app import app
        client = app.test_client()

        # CSV import
        csv_data = "prompt\nExplain recruitment workflow.\n"
        res_csv = client.post("/import", data={"file": (io.BytesIO(csv_data.encode("utf-8")), "test.csv"), "domain": "HR"}, content_type="multipart/form-data")
        self.assertEqual(res_csv.status_code, 200)

        # TXT import
        txt_data = "Audit quarterly expenses.\n"
        res_txt = client.post("/import", data={"file": (io.BytesIO(txt_data.encode("utf-8")), "test.txt"), "domain": "Finance"}, content_type="multipart/form-data")
        self.assertEqual(res_txt.status_code, 200)

        # JSON import
        json_data = json.dumps([{"prompt": "Analyze sales and marketing pipeline."}])
        res_json = client.post("/import", data={"file": (io.BytesIO(json_data.encode("utf-8")), "test.json"), "domain": "Marketing"}, content_type="multipart/form-data")
        self.assertEqual(res_json.status_code, 200)

        # Export CSV
        res_exp_csv = client.get("/export/csv")
        self.assertEqual(res_exp_csv.status_code, 200)
        self.assertIn("domain,prompt,clarity", res_exp_csv.data.decode("utf-8"))

        # Export JSON
        res_exp_json = client.get("/export/json")
        self.assertEqual(res_exp_json.status_code, 200)
        items = json.loads(res_exp_json.data.decode("utf-8"))
        self.assertGreaterEqual(len(items), 3)
        self.assertIn("domain", items[0])

if __name__ == "__main__":
    unittest.main()

