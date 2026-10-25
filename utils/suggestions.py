"""
Suggestion Engine & Deterministic Prompt Improvement Module.
Generates explainable, rule-based suggestions and deterministic improved prompts
without calling any external LLM APIs.
"""
import re
from utils.feature_extraction import extract_features

def generate_suggestions(prompt: str, domain: str = "General", dimensions: dict = None) -> list:
    """
    Generates rule-based improvement suggestions based on extracted features, scores,
    and the selected enterprise domain.
    """
    feats = extract_features(prompt, domain=domain)
    suggestions = []

    # 1. Length check
    if feats["word_count"] < 8:
        suggestions.append("Add more details to your prompt.")

    # 2. Instruction check
    if feats["instruction_words"] == 0:
        suggestions.append("Clearly mention what you want the system to do (e.g., analyze, explain, design, compare).")

    # 3. Domain relevance check
    if domain == "HR" and feats["domain_relevance"] == 0:
        suggestions.append("Add HR-specific context such as job role, candidate experience level, or performance evaluation criteria.")
    elif domain == "Finance" and feats["domain_relevance"] == 0:
        suggestions.append("Add finance-specific parameters such as fiscal timeframe, budget numbers, expense categories, or reporting standards.")
    elif domain == "Marketing" and feats["domain_relevance"] == 0:
        suggestions.append("Add marketing-specific details such as target audience persona, campaign channel, value proposition, or call-to-action.")

    # 4. Target audience check
    if feats["audience_indicators"] == 0:
        suggestions.append("Specify the target audience (e.g., beginners, college students, executives).")

    # 5. Context check
    if feats["context_indicators"] == 0:
        suggestions.append("Provide more context (e.g., background scenario, domain setting, or project scope).")

    # 6. Output format check
    if feats["format_indicators"] == 0:
        suggestions.append("Specify the expected output format (e.g., bullet points, markdown table, step-by-step list).")

    # 7. Specificity check
    if feats["specificity_keywords"] == 0 or feats["example_indicators"] == 0:
        suggestions.append("Add specific requirements, examples, or constraints.")

    # If the prompt is already very complete
    if not suggestions:
        suggestions.append(f"Prompt is well-structured and aligns well with {domain} enterprise quality standards.")

    return suggestions

def generate_improved_prompt(prompt: str, domain: str = "General", overall_score: float = 50.0) -> str:
    """
    Synthesizes an improved prompt deterministically using structured domain templates.
    Never calls external LLMs.
    """
    clean = prompt.strip().rstrip(".")
    if not clean:
        return f"Please provide a detailed prompt describing your desired {domain} task and requirements."

    # If already high quality (>= 80.0, Excellent), return original clean prompt
    if overall_score >= 80.0:
        return clean

    lower = clean.lower()

    # Determine core action verb
    action = "Analyze and structure"
    verbs = ["explain", "describe", "analyze", "summarize", "write", "create", "compare", "evaluate", "design", "build", "draft"]
    for v in verbs:
        if re.search(rf"\b{v}\b", lower):
            action = v.capitalize()
            break

    # Strip existing action verb to get clean subject
    subject = re.sub(r"^(?:please\s+|can you\s+|i want to\s+|write\s+a\s+|explain\s+|tell me about\s+)", "", clean, flags=re.IGNORECASE).strip()
    if not subject:
        subject = clean

    # Domain-specific enterprise templates
    if domain == "HR":
        improved = (
            f"Act as an Enterprise HR Director. {action} {subject} for workforce and talent management. "
            f"Specify target roles, clear competency criteria, and format the output as a structured rubric "
            f"with actionable steps."
        )
    elif domain == "Finance":
        improved = (
            f"Act as a Corporate Financial Controller. {action} {subject} for executive leadership. "
            f"Include fiscal metrics, budget breakdown, variance analysis, and format key findings "
            f"in a structured markdown summary table."
        )
    elif domain == "Marketing":
        improved = (
            f"Act as a Senior Growth Marketing Lead. {action} {subject} for our target customer demographic. "
            f"Include campaign objectives, key value propositions, conversion channels, and format "
            f"as an actionable marketing plan with copy examples."
        )
    else:  # General
        improved = (
            f"Act as an Enterprise Operations Lead. {action} {subject} for organizational stakeholders. "
            f"Provide a step-by-step breakdown covering key objectives, structured guidelines, and practical examples "
            f"formatted with a summary checklist."
        )

    return improved
