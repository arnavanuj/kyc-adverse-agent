from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request

import nltk
from nltk.tokenize import PunktSentenceTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from app.tools.prompt_store import load_prompt_text

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional at runtime with TF-IDF fallback
    SentenceTransformer = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
COMPRESSION_MODEL = "phi3:mini"
CLASSIFICATION_MODEL = "mistral:7b"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K_RELEVANT_CHUNKS = 3
TOP_K_SENTENCES_PER_CHUNK = 3
GLOBAL_TOP_EVIDENCE_CHUNKS = 2

embedding_model: object | None = None
_embedding_model_failed = False

DEFAULT_REASONING_GUIDANCE = (
    "Analyze the evidence below and determine if it contains adverse media."
)
CLASSIFICATION_OUTPUT_SCHEMA = """Return STRICT JSON with the following fields:
- risk_labels (list)
- risk_score (0 to 1)
- rationale
- evidence_snippets (list)
- sources (list)"""

DEFAULT_COMPRESSION_GUIDANCE = (
    "Extract the key adverse media facts about the subject from the text below."
)
DEFAULT_EVIDENCE_SELECTION_GUIDANCE = (
    "Select evidence sentences that best represent adverse media allegations, "
    "prioritizing severe indicators when present."
)

LEGACY_COMPRESSION_PROMPT_PREFIX = """Extract the key adverse media facts about the subject from the text below.
Return 1-2 sentences only.

Subject:
{subject_query}

Text:
"""


def _get_embedding_model() -> object | None:
    global embedding_model, _embedding_model_failed
    if embedding_model is not None:
        return embedding_model
    if _embedding_model_failed:
        return None
    if SentenceTransformer is None:
        _embedding_model_failed = True
        logger.warning("sentence-transformers import failed, using TF-IDF fallback")
        return None
    try:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        return embedding_model
    except Exception as exc:
        _embedding_model_failed = True
        logger.warning("Embedding model unavailable, using TF-IDF fallback: %s", exc)
        return None


def _load_stage_guidance(stage: str, fallback: str) -> str:
    prompt_text = load_prompt_text(stage).strip()
    return prompt_text if prompt_text else fallback


def _default_finding(article: dict, rationale: str = "No major risk indicators detected.") -> dict:
    return {
        "url": article.get("url", ""),
        "title": article.get("title", ""),
        "risk_labels": [],
        "risk_score": 0.0,
        "confidence": 0.15,
        "rationale": rationale,
        "evidence_snippets": [],
    }


def _default_findings_for_articles(articles: list[dict], rationale: str) -> list[dict]:
    return [_default_finding(article, rationale=rationale) for article in articles]


def _clamp_score(value: object) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, score))


def _extract_json_payload(text: str) -> dict | None:
    if not text:
        return None

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    json_match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not json_match:
        return None

    try:
        parsed = json.loads(json_match.group(0))
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        return None

    return None


def _normalize_list(items: object) -> list[str]:
    if not isinstance(items, list):
        return []

    normalized: list[str] = []
    for item in items:
        if isinstance(item, str):
            trimmed = item.strip()
            if trimmed:
                normalized.append(trimmed)
    return normalized


def split_article_into_chunks(article_text: str, max_words: int = 1500) -> list[str]:
    words = article_text.split()
    if not words:
        return []
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


def _ollama_generate_text(prompt: str, model: str) -> str:
    logger.info("OLLAMA CALL -> model=%s prompt_length=%s", model, len(prompt))
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }

    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        OLLAMA_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=300) as response:
        response_body = response.read().decode("utf-8")
    logger.info("OLLAMA RESPONSE LENGTH=%s", len(response_body))

    outer = json.loads(response_body)
    return str(outer.get("response", "")).strip()


def select_top_semantic_chunks(
    chunks: list[str], user_query: str, top_k: int = TOP_K_RELEVANT_CHUNKS
) -> tuple[list[str], list[float]]:
    if not chunks:
        return [], []

    query = user_query.strip() or "Unknown subject"
    chunk_text = chunks[0]
    logger.info("EMBEDDING INPUT CHUNK PREVIEW:\n%s", chunk_text[:500])
    logger.info("STEP 5: Generating embeddings for %d chunks", len(chunks))
    model = _get_embedding_model()
    if model is not None:
        chunk_embeddings = model.encode(chunks)
        query_embedding = model.encode(query)
        logger.info("Query embedding sample: %s", query_embedding[:5])
    else:
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform([query, *chunks]).toarray()
        query_embedding = matrix[0]
        chunk_embeddings = matrix[1:]

    logger.info("STEP 6: Computing cosine similarity for query: %s", query)
    scores = cosine_similarity([query_embedding], chunk_embeddings)[0]
    logger.info("Similarity scores: %s", scores.tolist())

    logger.info("STEP 7: Selecting top %d chunks", TOP_K_RELEVANT_CHUNKS)
    top_count = min(top_k, len(chunks))
    top_indices = scores.argsort()[-top_count:][::-1]
    relevant_chunks = [chunks[i] for i in top_indices]
    top_scores = [float(scores[i]) for i in top_indices]
    logger.info("Top similarity scores: %s", top_scores)

    return relevant_chunks, top_scores


def select_top_sentences_from_chunk(
    chunk_text: str, query_embedding: object, top_k_sentences: int = TOP_K_SENTENCES_PER_CHUNK
) -> tuple[list[str], list[float]]:
    tokenizer = PunktSentenceTokenizer()
    sentences = [s.strip() for s in tokenizer.tokenize(chunk_text) if s.strip()]
    if not sentences:
        return [], []

    logger.info("STEP 7B: sentence filtering")
    logger.info("Total sentences: %d", len(sentences))

    if isinstance(query_embedding, str):
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform([query_embedding, *sentences]).toarray()
        query_vector = matrix[0]
        sentence_embeddings = matrix[1:]
        scores = cosine_similarity([query_vector], sentence_embeddings)[0]
    else:
        model = _get_embedding_model()
        if model is None:
            vectorizer = TfidfVectorizer()
            matrix = vectorizer.fit_transform(["Unknown subject", *sentences]).toarray()
            query_vector = matrix[0]
            sentence_embeddings = matrix[1:]
            scores = cosine_similarity([query_vector], sentence_embeddings)[0]
        else:
            sentence_embeddings = model.encode(sentences)
            scores = cosine_similarity([query_embedding], sentence_embeddings)[0]

    top_count = min(top_k_sentences, len(sentences))
    top_indices = scores.argsort()[-top_count:][::-1]
    top_sentences = [sentences[i] for i in top_indices]
    top_scores = [float(scores[i]) for i in top_indices]

    logger.info("Top sentences selected: %d", len(top_sentences))
    logger.info("Top similarity scores: %s", [round(score, 4) for score in top_scores])
    logger.info("Sentence previews: %s", [sentence[:180] for sentence in top_sentences])
    return top_sentences, top_scores


def _collect_article_evidence(article: dict, user_query: str, query_embedding: object) -> list[dict]:
    clean_text = (article.get("content") or "").strip()
    if not clean_text:
        return []
    logger.info("SCRAPED CLEAN TEXT PREVIEW:\n%s", clean_text[:1000])

    logger.info("STEP 4: Splitting article into chunks")
    chunks = split_article_into_chunks(clean_text, max_words=1500)
    logger.info("Article split into %d chunks", len(chunks))
    if not chunks:
        return []

    relevant_chunks, top_scores = select_top_semantic_chunks(chunks, user_query, top_k=TOP_K_RELEVANT_CHUNKS)

    evidence_items: list[dict] = []
    for chunk, score in zip(relevant_chunks, top_scores):
        top_sentences, sentence_scores = select_top_sentences_from_chunk(
            chunk,
            query_embedding,
            top_k_sentences=TOP_K_SENTENCES_PER_CHUNK,
        )
        if not top_sentences:
            continue
        evidence_items.append(
            {
                "url": article.get("url", ""),
                "title": article.get("title", ""),
                "sentences": top_sentences,
                "similarity_scores": sentence_scores,
                "similarity_score": float(score),
                "chunk_char_count": len(chunk),
            }
        )
    return evidence_items


def _build_compression_prompt(user_query: str, evidence_item: dict) -> str:
    guidance = _load_stage_guidance("compression_prompt", DEFAULT_COMPRESSION_GUIDANCE)
    sentences = evidence_item.get("sentences", [])
    sentence_block = "\n".join(f"- {sentence}" for sentence in sentences if sentence.strip())
    if not sentence_block:
        sentence_block = "- No relevant sentences extracted."
    return "\n\n".join(
        [
            guidance,
            "Return 1-2 sentences only.",
            f"Subject:\n{user_query.strip() or 'Unknown subject'}",
            f"Evidence sentences:\n{sentence_block}",
            "Task:\nSummarize whether these sentences contain adverse media about the subject.",
        ]
    )


def _compress_evidence_item(evidence_item: dict, user_query: str, prompt: str | None = None) -> dict:
    prompt = prompt or _build_compression_prompt(user_query, evidence_item)
    response = _ollama_generate_text(prompt, model=COMPRESSION_MODEL).strip()
    cleaned = re.sub(r"^```(?:text|markdown)?\s*", "", response, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    if not cleaned:
        cleaned = " ".join(evidence_item.get("sentences", []))[:280]

    compressed = dict(evidence_item)
    compressed["compressed_fact"] = cleaned
    return compressed


def _build_compressed_context(compressed_evidence: list[dict]) -> str:
    packed_parts: list[str] = []
    for index, item in enumerate(compressed_evidence, start=1):
        packed_parts.append(
            "\n".join(
                [
                    f"EVIDENCE {index}",
                    f"Source: {item.get('url', '')}",
                    f"Title: {item.get('title', '')}",
                    f"Fact: {item.get('compressed_fact', '')}",
                ]
            )
        )
    return "\n\n".join(packed_parts)


def _build_classification_prompt(packed_context: str) -> str:
    reasoning_guidance = _load_stage_guidance("reasoning_prompt", DEFAULT_REASONING_GUIDANCE)
    return "\n\n".join(
        [
            "You are an AML compliance analyst.",
            reasoning_guidance,
            CLASSIFICATION_OUTPUT_SCHEMA,
            f"Evidence:\n{packed_context}",
        ]
    )


def _build_evidence_selection_query(subject_query: str) -> str:
    evidence_guidance = _load_stage_guidance("evidence_selection_prompt", DEFAULT_EVIDENCE_SELECTION_GUIDANCE)
    return "\n\n".join([subject_query.strip() or "Unknown subject", evidence_guidance])


def _ollama_classify_aggregated_context(packed_context: str) -> dict:
    prompt = _build_classification_prompt(packed_context)
    llm_text = _ollama_generate_text(prompt, model=CLASSIFICATION_MODEL)
    logger.info("STEP 10: Final classification received")
    logger.info("LLM response length: %d", len(llm_text))

    parsed = _extract_json_payload(llm_text)
    if parsed is None:
        raise ValueError("Ollama returned malformed JSON content")

    return parsed


async def classify_many(articles: list[dict], user_query: str = "") -> list[dict]:
    logger.info("Invoking LLM risk classifier for %d articles", len(articles))
    if not articles:
        return []

    try:
        query_text = user_query.strip() or "Unknown subject"
        evidence_selection_query = _build_evidence_selection_query(query_text)
        model = _get_embedding_model()
        if model is None:
            query_embedding: object = evidence_selection_query
            logger.info("Embedding fallback enabled: using query text for TF-IDF scoring")
        else:
            query_embedding = model.encode(evidence_selection_query)
            logger.info("Embedding optimization: query embedding computed once and reused across chunks")

        all_evidence: list[dict] = []
        for article in articles:
            all_evidence.extend(_collect_article_evidence(article, evidence_selection_query, query_embedding))

        logger.info("STEP 8A: Aggregating evidence across articles")
        ranked_evidence = sorted(all_evidence, key=lambda item: item.get("similarity_score", 0.0), reverse=True)
        top_evidence = ranked_evidence[:GLOBAL_TOP_EVIDENCE_CHUNKS]

        logger.info("STEP 8B: Selected top evidence chunks")
        logger.info(
            "Global top similarity scores: %s",
            [round(item.get("similarity_score", 0.0), 4) for item in top_evidence],
        )

        if not top_evidence:
            return _default_findings_for_articles(
                articles,
                rationale="No relevant chunks selected for aggregated classification.",
            )

        logger.info("STEP 8C: Compressing evidence using small model")
        compressed_evidence: list[dict] = []
        legacy_prompt_overhead = len(
            LEGACY_COMPRESSION_PROMPT_PREFIX.format(subject_query=user_query.strip() or "Unknown subject")
        )
        prompt_lengths_before: list[int] = []
        prompt_lengths_after: list[int] = []
        compression_durations: list[float] = []
        for index, item in enumerate(top_evidence):
            try:
                logger.info("STEP 8C: compressing evidence chunk %s/%s", index + 1, len(top_evidence))
                compression_prompt = _build_compression_prompt(user_query, item)
                prompt_char_length = len(compression_prompt)
                prompt_token_estimate = max(1, prompt_char_length // 4)
                logger.info("STEP 8C: prompt char length=%s", prompt_char_length)
                logger.info("STEP 8C: prompt token length approx=%s", prompt_token_estimate)
                prompt_lengths_after.append(prompt_char_length)
                prompt_lengths_before.append(legacy_prompt_overhead + int(item.get("chunk_char_count", 0)))
                start = time.time()
                result = await asyncio.to_thread(_compress_evidence_item, item, user_query, compression_prompt)
                compressed_evidence.append(result)
                compression_durations.append(time.time() - start)
                logger.info("STEP 8C: compression finished for chunk %s", index + 1)
            except Exception as exc:
                logger.warning("Evidence compression failed for item %d: %s", index + 1, exc)
                fallback_item = dict(item)
                fallback_item["compressed_fact"] = " ".join(fallback_item.get("sentences", []))[:280]
                compressed_evidence.append(fallback_item)

        logger.info("STEP 8D: building compressed context with %s chunks", len(compressed_evidence))
        packed_context = _build_compressed_context(compressed_evidence)
        logger.info("STEP 8D context preview:\n%s", packed_context[:500])
        logger.info("STEP 9: Sending compressed evidence to reasoning model")
        classification_prompt = _build_classification_prompt(packed_context)
        logger.info("STEP 9 prompt length characters=%s", len(classification_prompt))
        start = time.time()
        llm_result = await asyncio.to_thread(_ollama_classify_aggregated_context, packed_context)
        reasoning_duration = time.time() - start
        logger.info("STEP 9 reasoning completed in %.2f seconds", reasoning_duration)

        if top_evidence and prompt_lengths_after and prompt_lengths_before:
            avg_chunk_size = sum(int(item.get("chunk_char_count", 0)) for item in top_evidence) / len(top_evidence)
            avg_sentence_count = sum(len(item.get("sentences", [])) for item in top_evidence) / len(top_evidence)
            avg_prompt_before = sum(prompt_lengths_before) / len(prompt_lengths_before)
            avg_prompt_after = sum(prompt_lengths_after) / len(prompt_lengths_after)
            reduction_pct = 0.0
            if avg_prompt_before > 0:
                reduction_pct = ((avg_prompt_before - avg_prompt_after) / avg_prompt_before) * 100
            avg_compression_duration = 0.0
            if compression_durations:
                avg_compression_duration = sum(compression_durations) / len(compression_durations)

            logger.info("PERF: average chunk size before filtering=%.2f chars", avg_chunk_size)
            logger.info("PERF: average sentence count after filtering=%.2f", avg_sentence_count)
            logger.info(
                "PERF: average compression prompt chars before=%.2f after=%.2f reduction=%.2f%%",
                avg_prompt_before,
                avg_prompt_after,
                reduction_pct,
            )
            logger.info(
                "PERF: LLM inference duration compression_avg=%.2fs reasoning=%.2fs",
                avg_compression_duration,
                reasoning_duration,
            )

        labels = _normalize_list(llm_result.get("risk_labels"))
        score = _clamp_score(llm_result.get("risk_score"))
        rationale = str(llm_result.get("rationale", "")).strip() or "No major risk indicators detected."
        llm_snippets = _normalize_list(llm_result.get("evidence_snippets"))[:5]
        llm_sources = _normalize_list(llm_result.get("sources"))

        evidence_by_url: dict[str, list[str]] = {}
        for item in compressed_evidence:
            url = item.get("url", "")
            if not url:
                continue
            evidence_by_url.setdefault(url, []).append(str(item.get("compressed_fact", "")))

        selected_urls = set(evidence_by_url.keys())
        if llm_sources:
            matched_urls = {
                url
                for url in evidence_by_url
                if any((src in url) or (url in src) for src in llm_sources)
            }
            if matched_urls:
                selected_urls = matched_urls

        findings: list[dict] = []
        for article in articles:
            url = article.get("url", "")
            if url in selected_urls:
                evidence = llm_snippets or evidence_by_url.get(url, [])[:3]
                confidence = 0.15 if score == 0 else min(0.95, 0.35 + (score * 0.6))
                findings.append(
                    {
                        "url": url,
                        "title": article.get("title", ""),
                        "risk_labels": labels,
                        "risk_score": round(score, 3),
                        "confidence": round(confidence, 3),
                        "rationale": rationale,
                        "evidence_snippets": evidence,
                    }
                )
            else:
                findings.append(
                    _default_finding(
                        article,
                        rationale="No globally ranked evidence selected for this article.",
                    )
                )

        return findings

    except (json.JSONDecodeError, ValueError, urllib.error.URLError, TimeoutError) as exc:
        logger.warning(
            "Aggregated LLM classification failed. Falling back to zero risk. Error: %s",
            exc,
        )
        return _default_findings_for_articles(
            articles,
            rationale="LLM classifier unavailable or malformed response.",
        )
    except Exception:
        logger.exception("Unexpected aggregated LLM classification error. Falling back to zero risk.")
        return _default_findings_for_articles(
            articles,
            rationale="LLM classifier unavailable or malformed response.",
        )


def aggregate_risk(findings: list[dict]) -> tuple[float, str]:
    if not findings:
        return 0.0, "low"

    score = sum(item.get("risk_score", 0.0) for item in findings) / len(findings)

    if score >= 0.65:
        return round(score, 3), "high"
    if score >= 0.35:
        return round(score, 3), "medium"
    return round(score, 3), "low"
