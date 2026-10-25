

"""
Text Preprocessing Module.
Provides simple, explainable text cleaning, tokenization, and sentence splitting.
"""
import re

def clean_text(text: str) -> str:
    """Removes irregular whitespace and normalizes text."""
    if not text:
        return ""
    # Normalize multiple whitespace, tabs, and newlines to single space
    cleaned = re.sub(r"\s+", " ", text).strip()
    return cleaned

def tokenize_words(text: str) -> list:
    """Extracts lowercase words from text."""
    cleaned = clean_text(text)
    if not cleaned:
        return []
    words = re.findall(r"\b[a-zA-Z0-9_\-']+\b", cleaned.lower())
    return words

def split_sentences(text: str) -> list:
    """Splits text into sentences based on punctuation."""
    cleaned = clean_text(text)
    if not cleaned:
        return []
    sentences = re.split(r"[.!?]+(?:\s+|$)", cleaned)
    return [s.strip() for s in sentences if s.strip()]
