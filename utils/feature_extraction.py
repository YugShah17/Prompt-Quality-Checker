"""
Feature Extraction Module for Prompt Quality Assessment.
Extracts 13 explainable, interpretable linguistic, structural, and domain features
from raw prompt text for Machine Learning model training and inference.
"""
import re
from utils.preprocessing import clean_text, tokenize_words, split_sentences

# 1. Action / Instruction Verbs
INSTRUCTION_WORDS = {
    "analyze", "explain", "describe", "evaluate", "compare", "summarize",
    "generate", "create", "write", "develop", "identify", "calculate",
    "implement", "outline", "design", "build", "review", "synthesize",
    "formulate", "assess", "structure", "draft", "discuss", "solve", "list"
}

# 2. Specificity Indicators (numbers, percentages, metrics, technical terms)
SPECIFICITY_KEYWORDS = {
    "specifically", "metrics", "benchmark", "parameters", "dataset",
    "python", "sql", "api", "function", "algorithm", "database",
    "architecture", "protocol", "step-by-step", "workflow", "methodology",
    "numerical", "percentage", "accuracy", "latency", "budget"
}

# 3. Context Indicators
CONTEXT_KEYWORDS = {
    "background", "context", "scenario", "situation", "project",
    "enterprise", "company", "organization", "industry", "environment",
    "system", "production", "infrastructure", "quarterly", "framework"
}

# 4. Target Audience Indicators
AUDIENCE_KEYWORDS = {
    "audience", "beginner", "beginners", "student", "students",
    "undergraduate", "developer", "developers", "engineer", "engineers",
    "executive", "executives", "manager", "managers", "leadership",
    "stakeholder", "stakeholders", "client", "clients", "user", "users",
    "team", "non-technical", "junior", "senior", "c-level"
}

# 5. Output Format Indicators
FORMAT_KEYWORDS = {
    "table", "json", "markdown", "bullet", "bullets", "points",
    "list", "numbered", "schema", "format", "checklist", "structure",
    "template", "code", "summary", "headings", "overview", "report"
}

# 6. Constraint Indicators
CONSTRAINT_KEYWORDS = {
    "constraint", "constraints", "limit", "within", "maximum",
    "minimum", "avoid", "ensure", "strictly", "words", "tone",
    "formal", "concise", "under", "between", "do not", "without"
}

# 7. Example / Reference Indicators
EXAMPLE_KEYWORDS = {
    "example", "examples", "sample", "samples", "e.g.", "such as",
    "instance", "reference", "illustration", "case", "demonstration"
}

# 8. Enterprise Domain Specific Lexicons
DOMAIN_KEYWORDS = {
    "HR": {
        "recruitment", "employee", "employees", "hiring", "interview", "interviews",
        "onboarding", "talent", "staff", "workforce", "pip", "retention", "attrition",
        "training", "performance", "resume", "job", "candidate", "candidates",
        "appraisal", "compensation", "benefits", "culture", "payroll", "mentorship",
        "leave", "promotion", "rubric", "workplace", "turnover", "peer", "conduct"
    },
    "Finance": {
        "finance", "financial", "budget", "budgets", "budgeting", "revenue",
        "expense", "expenses", "profit", "margin", "margins", "forecast", "forecasting",
        "audit", "auditing", "dcf", "valuation", "cost", "costs", "spend", "roi",
        "variance", "fiscal", "quarterly", "q1", "q2", "q3", "q4", "fy24", "fy25", "fy26",
        "tax", "balance", "cash", "asset", "debt", "investment", "inr", "usd", "dollar",
        "accounting", "ledger", "capex", "opex", "pricing", "ebitda"
    },
    "Marketing": {
        "marketing", "campaign", "campaigns", "advertising", "ad", "ads",
        "customer", "customers", "client", "clients", "conversion", "conversions",
        "seo", "content", "brand", "branding", "lead", "leads", "traffic", "social",
        "media", "churn", "engagement", "copywriting", "funnel", "click-through",
        "ctr", "acquisition", "webinar", "launch", "buyer", "buyers", "audience",
        "promote", "promotional", "newsletter", "headline", "rebranding", "nps"
    },
    "General": {
        "enterprise", "operation", "operational", "process", "workflow", "project",
        "system", "guideline", "guidelines", "standards", "management", "strategy",
        "review", "objective", "analysis", "report", "overview", "plan", "checklist",
        "roadmap", "summary", "architecture", "data", "technical", "engineering"
    }
}

FEATURE_NAMES = [
    "word_count",
    "char_count",
    "sentence_count",
    "question_marks",
    "instruction_words",
    "specificity_keywords",
    "context_indicators",
    "audience_indicators",
    "format_indicators",
    "constraint_indicators",
    "example_indicators",
    "avg_word_length",
    "domain_relevance"
]

def extract_features(prompt: str, domain: str = "General") -> dict:
    """
    Extracts a dictionary of explainable features from a user prompt,
    including domain alignment score for the selected enterprise domain.
    """
    text = clean_text(prompt)
    text_lower = text.lower()
    words = tokenize_words(text)
    sentences = split_sentences(text)

    word_count = len(words)
    char_count = len(text)
    sentence_count = max(1, len(sentences))
    avg_word_length = round(sum(len(w) for w in words) / word_count, 2) if word_count > 0 else 0.0

    # Question mark count
    question_marks = text.count("?")

    # Count instruction words
    instruction_words = sum(1 for w in words if w in INSTRUCTION_WORDS)

    # Count specificity keywords and numeric patterns
    has_digits = len(re.findall(r"\b\d+(\.\d+)?\b", text))
    has_pct = len(re.findall(r"\b\d+%", text))
    spec_word_hits = sum(1 for w in words if w in SPECIFICITY_KEYWORDS)
    specificity_keywords = spec_word_hits + min(5, has_digits) + min(3, has_pct)

    # Count context indicators
    context_indicators = sum(1 for w in words if w in CONTEXT_KEYWORDS)

    # Count audience indicators
    audience_indicators = sum(1 for w in words if w in AUDIENCE_KEYWORDS)
    if "for " in text_lower or "target" in text_lower:
        audience_indicators += 1

    # Count output format indicators
    format_indicators = sum(1 for w in words if w in FORMAT_KEYWORDS)

    # Count constraint indicators
    constraint_indicators = sum(1 for w in words if w in CONSTRAINT_KEYWORDS)

    # Count example / reference indicators
    example_indicators = sum(1 for w in words if w in EXAMPLE_KEYWORDS)
    if "e.g." in text_lower or "such as" in text_lower:
        example_indicators += 1

    # Domain relevance feature
    norm_domain = domain if domain in DOMAIN_KEYWORDS else "General"
    domain_lexicon = DOMAIN_KEYWORDS.get(norm_domain, DOMAIN_KEYWORDS["General"])
    domain_hits = sum(1 for w in words if w in domain_lexicon)
    domain_relevance = min(5, domain_hits)

    return {
        "word_count": word_count,
        "char_count": char_count,
        "sentence_count": sentence_count,
        "question_marks": question_marks,
        "instruction_words": instruction_words,
        "specificity_keywords": specificity_keywords,
        "context_indicators": context_indicators,
        "audience_indicators": audience_indicators,
        "format_indicators": format_indicators,
        "constraint_indicators": constraint_indicators,
        "example_indicators": example_indicators,
        "avg_word_length": avg_word_length,
        "domain_relevance": domain_relevance
    }

def get_feature_vector(prompt: str, domain: str = "General") -> list:
    """Extracts features and returns an ordered list matching FEATURE_NAMES."""
    feats = extract_features(prompt, domain=domain)
    return [feats[name] for name in FEATURE_NAMES]
