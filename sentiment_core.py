"""
User review sentiment — Doc2Vec document vectors + k-NN classifier.

Follows Joshua Eckroth (Packt Ch. 3): preprocess review text, infer vectors with
Doc2Vec, classify positive (1) vs negative (0) with KNeighborsClassifier (k=9).
Training uses the Kotzias et al. labelled sentence files (Yelp, Amazon, IMDb).
"""

from __future__ import annotations

import pickle
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

DATASET_NAMES = ("yelp", "amazon_cells", "imdb")
K_NEIGHBORS = 9
DOC2VEC_VECTOR_SIZE = 50
DOC2VEC_EPOCHS = 30
INFER_EPOCHS = 10
RANDOM_SEED = 42


def default_data_dir() -> Path:
    return Path(__file__).resolve().parent / "data" / "sentiment_labelled_sentences"


def default_cache_dir() -> Path:
    return Path(__file__).resolve().parent / "web" / "data" / "cache"


def extract_words(sentence: str) -> list[str]:
    """Tokenize like the book's updated preprocessing (SentimentAnalysisnew.ipynb)."""
    sent = sentence.lower()
    sent = re.sub(r"<[^>]+>", " ", sent)
    sent = re.sub(r"(\w)'(\w)", r"\1\2", sent)
    sent = re.sub(r"\W", " ", sent)
    sent = re.sub(r"\s+", " ", sent)
    sent = sent.strip()
    return sent.split() if sent else []


def load_labelled_sentences(data_dir: Path | None = None) -> tuple[list[str], list[int]]:
    data_dir = data_dir or default_data_dir()
    texts: list[str] = []
    labels: list[int] = []
    for fname in DATASET_NAMES:
        path = data_dir / f"{fname}_labelled.txt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing labelled data file: {path}")
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split("\t")
                if len(parts) != 2:
                    continue
                texts.append(parts[0])
                labels.append(int(parts[1]))
    return texts, labels


def _build_tagged_documents(texts: list[str]) -> list[TaggedDocument]:
    return [
        TaggedDocument(extract_words(text), [f"doc_{index}"])
        for index, text in enumerate(texts)
    ]


def train_doc2vec(documents: list[TaggedDocument]) -> Doc2Vec:
    random.seed(RANDOM_SEED)
    model = Doc2Vec(
        vector_size=DOC2VEC_VECTOR_SIZE,
        min_count=1,
        dm=0,
        hs=1,
        seed=RANDOM_SEED,
    )
    model.build_vocab(documents)
    model.train(
        documents,
        total_examples=model.corpus_count,
        epochs=DOC2VEC_EPOCHS,
    )
    return model


def vectors_for_texts(model: Doc2Vec, texts: list[str]) -> np.ndarray:
    matrix = [
        model.infer_vector(extract_words(text), epochs=INFER_EPOCHS)
        for text in texts
    ]
    return np.array(matrix)


@dataclass
class SentimentPipeline:
    doc2vec: Doc2Vec
    classifier: KNeighborsClassifier
    cv_mean_accuracy: float | None = None

    def predict(self, text: str) -> dict[str, Any]:
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("Enter some review text.")
        words = extract_words(cleaned)
        if not words:
            raise ValueError("Review text has no usable words after preprocessing.")

        vector = self.doc2vec.infer_vector(words, epochs=INFER_EPOCHS)
        label = int(self.classifier.predict([vector])[0])
        probabilities = self.classifier.predict_proba([vector])[0]
        classes = list(self.classifier.classes_)
        prob_by_class = {int(c): float(p) for c, p in zip(classes, probabilities)}
        confidence = prob_by_class.get(label, max(probabilities))

        return {
            "label": label,
            "sentiment": "positive" if label == 1 else "negative",
            "confidence": round(confidence * 100, 1),
            "probabilities": {
                "negative": round(prob_by_class.get(0, 0.0) * 100, 1),
                "positive": round(prob_by_class.get(1, 0.0) * 100, 1),
            },
            "tokens": len(words),
        }


def train_pipeline(
    data_dir: Path | None = None,
    *,
    compute_cv: bool = True,
) -> SentimentPipeline:
    texts, labels = load_labelled_sentences(data_dir)
    documents = _build_tagged_documents(texts)
    doc2vec = train_doc2vec(documents)
    vectors = vectors_for_texts(doc2vec, texts)

    classifier = KNeighborsClassifier(n_neighbors=K_NEIGHBORS)
    classifier.fit(vectors, labels)

    cv_mean: float | None = None
    if compute_cv:
        scores = cross_val_score(classifier, vectors, labels, cv=5)
        cv_mean = round(float(np.mean(scores)) * 100, 1)

    return SentimentPipeline(
        doc2vec=doc2vec,
        classifier=classifier,
        cv_mean_accuracy=cv_mean,
    )


def save_pipeline(pipeline: SentimentPipeline, cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    pipeline.doc2vec.save(str(cache_dir / "reviews.doc2vec"))
    with (cache_dir / "knn.pkl").open("wb") as handle:
        pickle.dump(
            {
                "classifier": pipeline.classifier,
                "cv_mean_accuracy": pipeline.cv_mean_accuracy,
            },
            handle,
        )


def load_pipeline(cache_dir: Path) -> SentimentPipeline | None:
    model_path = cache_dir / "reviews.doc2vec"
    knn_path = cache_dir / "knn.pkl"
    if not model_path.is_file() or not knn_path.is_file():
        return None
    doc2vec = Doc2Vec.load(str(model_path))
    with knn_path.open("rb") as handle:
        payload = pickle.load(handle)
    return SentimentPipeline(
        doc2vec=doc2vec,
        classifier=payload["classifier"],
        cv_mean_accuracy=payload.get("cv_mean_accuracy"),
    )


def get_or_train_pipeline(
    cache_dir: Path | None = None,
    data_dir: Path | None = None,
    *,
    force_retrain: bool = False,
) -> SentimentPipeline:
    cache_dir = cache_dir or default_cache_dir()
    if not force_retrain:
        loaded = load_pipeline(cache_dir)
        if loaded is not None:
            return loaded
    pipeline = train_pipeline(data_dir=data_dir, compute_cv=True)
    save_pipeline(pipeline, cache_dir)
    return pipeline


def sample_reviews(data_dir: Path | None = None) -> Iterator[dict[str, str]]:
    """A few fixed examples for the web UI."""
    examples = [
        ("Highly recommended — great service and atmosphere.", "positive"),
        ("Waste of money; would not buy again.", "negative"),
        ("The food was tasty and staff were friendly.", "positive"),
    ]
    for text, sentiment in examples:
        yield {"text": text, "hint": sentiment}
