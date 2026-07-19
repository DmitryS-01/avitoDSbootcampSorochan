"""Semantic query retrieval with a local multilingual encoder."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

MODEL_NAME = "intfloat/multilingual-e5-small"
MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
CROSS_ENCODER_NAME = "DiTy/cross-encoder-russian-msmarco"
CROSS_ENCODER_REVISION = "9029bab08103ad171724b510d312befa5b476293"


def _encode(
    texts: Sequence[str],
    tokenizer: object,
    model: object,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """Encode query texts with mean pooling and L2 normalization."""
    embeddings: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(
            list(texts[start : start + batch_size]),
            max_length=192,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        batch = {name: value.to(device) for name, value in batch.items()}
        with torch.inference_mode():
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"][..., None].bool()
            hidden = hidden.masked_fill(~mask, 0.0)
            pooled = hidden.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        embeddings.append(pooled.cpu().numpy().astype(np.float32))
    return np.vstack(embeddings)


def semantic_query_scores(
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    labels: np.ndarray,
) -> np.ndarray:
    """Transfer labels from semantically and lexically similar queries."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    model = AutoModel.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    device = torch.device("cpu")
    model.to(device).eval()

    calibration_dense = _encode(
        [f"query: {text}" for text in calibration["query_text"].astype(str)],
        tokenizer,
        model,
        device,
    )
    test_dense = _encode(
        [f"query: {text}" for text in test["query_text"].astype(str)],
        tokenizer,
        model,
        device,
    )
    model.to("cpu")
    del model

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        sublinear_tf=True,
        max_features=80_000,
    )
    calibration_char = normalize(
        vectorizer.fit_transform(calibration["query_clean"]).tocsr()
    )
    test_char = normalize(vectorizer.transform(test["query_clean"]).tocsr())

    similarity = 0.85 * (test_dense @ calibration_dense.T)
    similarity += 0.15 * (test_char @ calibration_char.T).toarray()
    similarity = np.maximum(similarity, 0.0)
    neighbours = np.argpartition(-similarity, kth=19, axis=1)[:, :20]
    weights = np.take_along_axis(similarity, neighbours, axis=1) ** 8.0
    weight_matrix = np.zeros_like(similarity, dtype=np.float32)
    weight_matrix[np.arange(len(test))[:, None], neighbours] = weights

    scores = weight_matrix @ labels
    frequency = np.maximum(labels.sum(axis=0), 1.0)
    return scores / frequency[None, :] ** 0.3


def cross_encoder_scores(
    queries: pd.DataFrame,
    articles: pd.DataFrame,
    candidates: np.ndarray,
) -> np.ndarray:
    """Score the base top-10 with a Russian cross-encoder."""
    tokenizer = AutoTokenizer.from_pretrained(
        CROSS_ENCODER_NAME,
        revision=CROSS_ENCODER_REVISION,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        CROSS_ENCODER_NAME,
        revision=CROSS_ENCODER_REVISION,
    )
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device).eval()

    documents = [
        (
            f"{row.title_clean}. {row.title_clean}. {row.important_clean}. "
            f"{row.intro_clean}. {row.body_clean[:2500]}"
        )
        for row in articles.itertuples()
    ]
    query_texts = queries["query_clean"].tolist()
    flat_queries = np.repeat(query_texts, candidates.shape[1]).tolist()
    flat_columns = candidates.reshape(-1)
    flat_documents = [documents[column] for column in flat_columns]

    predictions: list[np.ndarray] = []
    batch_size = 16 if device.type == "mps" else 8
    for start in range(0, len(flat_queries), batch_size):
        batch = tokenizer(
            flat_queries[start : start + batch_size],
            flat_documents[start : start + batch_size],
            max_length=384,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        batch = {name: value.to(device) for name, value in batch.items()}
        with torch.inference_mode():
            logits = model(**batch).logits
        if logits.ndim == 2 and logits.shape[1] > 1:
            logits = logits[:, -1]
        predictions.append(logits.reshape(-1).float().cpu().numpy())

    scores = np.full(
        (len(queries), len(articles)),
        -1_000_000.0,
        dtype=np.float32,
    )
    rows = np.repeat(np.arange(len(queries)), candidates.shape[1])
    scores[rows, flat_columns] = np.concatenate(predictions).astype(np.float32)
    model.to("cpu")
    del model
    return scores
