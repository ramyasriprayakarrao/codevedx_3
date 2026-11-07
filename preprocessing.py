import re
from collections import Counter

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def tokenize(text):
    # A small standard-library tokenizer; no external NLP package is required.
    return [word for word in clean_text(text).split() if len(word) > 1]

def build_vocabulary(texts, min_frequency=1):
    counts = Counter()
    for text in texts:
        counts.update(tokenize(text))
    return sorted(word for word, count in counts.items() if count >= min_frequency)
