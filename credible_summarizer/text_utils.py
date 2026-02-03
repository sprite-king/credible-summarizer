import html
import re
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import jieba
except Exception:  # pragma: no cover - optional dependency
    jieba = None


def has_jieba() -> bool:
    return jieba is not None

from .models import Citation


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    if not text.strip():
        return []
    text = normalize_text(text)
    parts = re.split(r"(?<=[。！？!?；;])\s*|\n+", text)
    sentences = [p.strip() for p in parts if p.strip()]
    return sentences


def build_sentence_embeddings(sentences: List[str], use_jieba: bool = False) -> np.ndarray:
    if use_jieba and jieba is not None:
        tokenized = [tokenize_words(s, use_jieba=True) for s in sentences]
        vectorizer = TfidfVectorizer(
            tokenizer=lambda x: x,
            preprocessor=lambda x: x,
            token_pattern=None,
            min_df=1,
        )
        return vectorizer.fit_transform(tokenized)
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4))
    return vectorizer.fit_transform(sentences)


def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9']+", text)
    return [t.lower() if re.fullmatch(r"[A-Za-z0-9']+", t) else t for t in tokens]


def tokenize_words(text: str, use_jieba: bool = False) -> List[str]:
    if use_jieba and jieba is not None:
        tokens: List[str] = []
        for tok in jieba.cut(text, cut_all=False):
            tok = tok.strip()
            if not tok:
                continue
            tokens.append(tok.lower() if re.fullmatch(r"[A-Za-z0-9']+", tok) else tok)
        return tokens
    return tokenize(text)


def extract_keywords(sentences: List[str], top_k: int = 12, use_jieba: bool = False) -> List[str]:
    if not sentences:
        return []
    tokenized = [tokenize_words(s, use_jieba=use_jieba) for s in sentences]
    vectorizer = TfidfVectorizer(
        tokenizer=lambda x: x,
        preprocessor=lambda x: x,
        token_pattern=None,
        min_df=1,
    )
    tfidf = vectorizer.fit_transform(tokenized)
    scores = np.asarray(tfidf.mean(axis=0)).ravel()
    terms = np.array(vectorizer.get_feature_names_out())
    ranked = terms[np.argsort(-scores)]
    keywords = [t for t in ranked if len(t) >= 2][:top_k]
    return keywords


def highlight_text(text: str, keywords: List[str]) -> str:
    if not text or not keywords:
        return html.escape(text)
    escaped = html.escape(text)
    for kw in keywords:
        if not kw:
            continue
        escaped_kw = html.escape(kw)
        flags = re.IGNORECASE if re.fullmatch(r"[A-Za-z0-9']+", kw) else 0
        escaped = re.sub(
            re.escape(escaped_kw),
            lambda m: f"<mark>{m.group(0)}</mark>",
            escaped,
            flags=flags,
        )
    return escaped


def pagerank(similarity: np.ndarray, damping: float = 0.85, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    n = similarity.shape[0]
    if n == 0:
        return np.array([])
    sim = similarity.copy()
    np.fill_diagonal(sim, 0.0)
    row_sums = sim.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    transition = sim / row_sums

    rank = np.ones(n) / n
    for _ in range(max_iter):
        new_rank = (1 - damping) / n + damping * transition.T.dot(rank)
        if np.linalg.norm(new_rank - rank, ord=1) < tol:
            break
        rank = new_rank
    return rank


def textrank_summary(sentences: List[str], top_k: int, use_jieba: bool = False) -> List[int]:
    if not sentences:
        return []
    tfidf = build_sentence_embeddings(sentences, use_jieba=use_jieba)
    similarity = cosine_similarity(tfidf)
    scores = pagerank(similarity)
    ranked = np.argsort(-scores)
    selected = sorted(ranked[: top_k])
    return selected


def deduplicate_indices(sentences: List[str], indices: List[int], threshold: float, use_jieba: bool = False) -> List[int]:
    if not indices:
        return []
    tfidf = build_sentence_embeddings([sentences[i] for i in indices], use_jieba=use_jieba)
    similarity = cosine_similarity(tfidf)
    keep = []
    for i, _ in enumerate(indices):
        if all(similarity[i, j] < threshold for j, _ in enumerate(keep)):
            keep.append(i)
    return [indices[i] for i in keep]


def map_citations(
    sentences: List[str],
    summary_indices: List[int],
    top_n: int,
    use_jieba: bool = False,
) -> List[List[Citation]]:
    if not sentences:
        return []
    tfidf = build_sentence_embeddings(sentences, use_jieba=use_jieba)
    similarity = cosine_similarity(tfidf)
    citations = []
    for idx in summary_indices:
        sims = similarity[idx]
        ranked = np.argsort(-sims)
        picked = []
        for ridx in ranked:
            if len(picked) >= top_n:
                break
            picked.append(Citation(sentence=sentences[ridx], score=float(sims[ridx]), index=int(ridx)))
        citations.append(picked)
    return citations


def suggest_thresholds(sentences: List[str], use_jieba: bool = False) -> tuple[float, float]:
    if len(sentences) < 3:
        return 0.35, 0.05
    tfidf = build_sentence_embeddings(sentences, use_jieba=use_jieba)
    similarity = cosine_similarity(tfidf)
    np.fill_diagonal(similarity, -1.0)
    top1 = np.max(similarity, axis=1)
    top2 = np.partition(similarity, -2, axis=1)[:, -2]
    mask = top1 > 0
    top1 = top1[mask]
    top2 = top2[mask]
    if top1.size == 0:
        return 0.35, 0.05
    gaps = top1 - top2
    threshold = float(np.clip(np.percentile(top1, 30), 0.2, 0.6))
    gap = float(np.clip(np.percentile(gaps, 30), 0.02, 0.2))
    return threshold, gap


def format_score(score: float) -> str:
    return f"{score:.2f}"


def get_context(sentences: List[str], index: int) -> str:
    prev_sent = sentences[index - 1] if index - 1 >= 0 else ""
    next_sent = sentences[index + 1] if index + 1 < len(sentences) else ""
    parts = [s for s in [prev_sent, sentences[index], next_sent] if s]
    return " ".join(parts)
