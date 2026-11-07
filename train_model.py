import csv
import math
import pickle
from collections import Counter, defaultdict
from preprocessing import tokenize

DATA_FILE = "news.csv"
MODEL_FILE = "fake_news_model.pkl"

class NaiveBayesTextClassifier:
    """Multinomial Naive Bayes implemented from scratch (no scikit-learn)."""
    def __init__(self):
        self.labels = []
        self.vocabulary = set()
        self.log_prior = {}
        self.log_likelihood = {}

    def fit(self, texts, labels):
        if not texts or len(texts) != len(labels):
            raise ValueError("Training texts and labels must be non-empty and equal in length.")
        self.labels = sorted(set(labels))
        if len(self.labels) < 2:
            raise ValueError("Dataset must contain at least two classes: FAKE and REAL.")

        doc_counts = Counter(labels)
        word_counts = {label: Counter() for label in self.labels}
        totals = Counter()

        for text, label in zip(texts, labels):
            words = tokenize(text)
            word_counts[label].update(words)
            totals[label] += len(words)
            self.vocabulary.update(words)

        vocab_size = max(len(self.vocabulary), 1)
        total_docs = len(labels)
        for label in self.labels:
            self.log_prior[label] = math.log(doc_counts[label] / total_docs)
            denominator = totals[label] + vocab_size
            self.log_likelihood[label] = {
                word: math.log((word_counts[label][word] + 1) / denominator)
                for word in self.vocabulary
            }
        return self

    def predict_one(self, text):
        if not self.labels:
            raise RuntimeError("Model is not trained.")
        words = tokenize(text)
        scores = {}
        for label in self.labels:
            score = self.log_prior[label]
            likelihood = self.log_likelihood[label]
            score += sum(likelihood.get(word, math.log(1 / (len(self.vocabulary) + 1))) for word in words)
            scores[label] = score
        return max(scores, key=scores.get)

    def confidence(self, text):
        words = tokenize(text)
        scores = {}
        for label in self.labels:
            score = self.log_prior[label]
            likelihood = self.log_likelihood[label]
            score += sum(likelihood.get(word, math.log(1 / (len(self.vocabulary) + 1))) for word in words)
            scores[label] = score
        peak = max(scores.values())
        exp_scores = {k: math.exp(v - peak) for k, v in scores.items()}
        total = sum(exp_scores.values())
        label = max(exp_scores, key=exp_scores.get)
        return label, exp_scores[label] / total

def load_dataset(path=DATA_FILE):
    texts, labels = [], []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"text", "label"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("news.csv must contain columns named text and label.")
        for row in reader:
            text = (row.get("text") or "").strip()
            label = (row.get("label") or "").strip().upper()
            if text and label in {"FAKE", "REAL"}:
                texts.append(text)
                labels.append(label)
    if len(texts) < 4:
        raise ValueError("Add at least four labelled examples to news.csv.")
    return texts, labels

def train_and_save():
    texts, labels = load_dataset()
    model = NaiveBayesTextClassifier().fit(texts, labels)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    print(f"Training completed: {len(texts)} articles, {len(model.vocabulary)} unique words.")
    print(f"Saved model to {MODEL_FILE}")

if __name__ == "__main__":
    train_and_save()
