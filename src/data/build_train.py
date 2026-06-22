"""
Build the 1,000-example training dataset from Stack Overflow [docker] Q&A.

WHY THIS FILE EXISTS:
    Task #4 in the weekend plan: produce `data/train/train_1k.jsonl`. The
    training set MUST be disjoint from `data/eval/eval.jsonl` — if the model
    sees an eval question (or a paraphrase of one) during training, our
    benchmark becomes open-book and the numbers are worthless. This module
    encapsulates the four-stage pipeline: fetch → clean → dedup → write.

WHY STACK OVERFLOW:
    The eval set was hand-built from docs.docker.com pages. Picking SO for
    training keeps us in the same domain (Docker) while drawing from a
    completely different *kind* of source — real user questions with
    community-voted answers, rather than reference documentation. That helps
    a fine-tune learn to ANSWER QUESTIONS rather than recite docs.

PIPELINE STAGES (each is its own function below):
    1. fetch_so_questions   — REST calls to api.stackexchange.com
    2. clean_html           — strip <p>/<code>/<pre> noise from answer bodies
    3. dedup_against_eval   — embedding-based paraphrase filter (the disjoint
                              rule, enforced semantically not just by URL)
    4. write_train_jsonl    — final write with stable schema

NO CLI HERE. This module is imported by `scripts/03_build_train.py`. Keeping
the orchestration out of the library mirrors how src/eval is structured.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import requests
from bs4 import BeautifulSoup


# =============================================================================
# 1. SCHEMA
# =============================================================================
@dataclass
class TrainExample:
    """One row of the training set.

    Fields:
        question         : the SO question title (sometimes title + body if the
                           title alone is too terse).
        answer           : the accepted answer body, HTML-stripped to plain text.
        so_question_id   : SO numeric ID — provenance trail, also a stable dedup
                           key within the training set.
        score            : SO community vote count on the question. Kept so we
                           can later slice the dataset by quality if needed.
        source_url       : link back to the SO question page. Audit trail.
    """
    question:       str
    answer:         str
    so_question_id: int
    score:          int
    source_url:     str


# =============================================================================
# 2. FETCH from the Stack Exchange REST API
# =============================================================================
# API docs: https://api.stackexchange.com/docs
#
# WHY we don't authenticate:
#   Anonymous quota = 300 requests/IP/day. For 1,500 candidate Q&A pairs we
#   need ~30 requests total (15 question pages + 15 answer batches). Plenty
#   of headroom — no need to register an app + manage a client key.
#
# WHY two API calls, not one:
#   The default /questions endpoint returns metadata (id, title, score,
#   accepted_answer_id) but NOT the answer body. Adding body to that filter
#   would bloat each response. Faster + cleaner to:
#     (a) page through /questions to harvest accepted_answer_ids,
#     (b) batch-fetch those answer bodies via /answers/{ids}?filter=withbody.
SE_API_BASE = "https://api.stackexchange.com/2.3"


def _se_get(path: str, params: dict) -> dict:
    """Single GET to the SE API with quota-aware backoff.

    The API returns two relevant fields in the JSON envelope:
      - `backoff` : seconds we MUST wait before the next request from this IP.
                    Set when we're approaching the per-second rate limit.
      - `quota_remaining` : daily quota left. We log it but don't gate on it.

    We also enforce a tiny 0.1s sleep between calls so we never burst above
    the 30 req/s soft cap.
    """
    params = {**params, "site": "stackoverflow"}
    resp = requests.get(f"{SE_API_BASE}{path}", params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    backoff = payload.get("backoff", 0)
    if backoff:
        # The API politely tells us to slow down — comply or we get throttled.
        print(f"  [API] backoff requested: sleeping {backoff}s")
        time.sleep(backoff)
    time.sleep(0.1)
    return payload


def fetch_so_questions(target_n: int = 1500, tag: str = "docker") -> list[dict]:
    """Page through top-voted SO questions for `tag` until we have target_n.

    Returns a list of raw dicts straight from the SE API. We over-fetch
    (~1500 for a 1K final target) because the next stages drop rows:
      - filter_candidates removes unanswered / low-score / too-short rows
      - dedup_against_eval removes paraphrases of eval questions

    NOTE on `has_more`: the SE API's anonymous tier returns has_more=False
    even when more results exist on subsequent pages (verified empirically
    on 2026-06-21). We therefore IGNORE has_more and keep paginating until
    we either hit target_n or get an empty page. Each page burns 1 of the
    300 anonymous req/IP/day quota; 1500/100 = 15 pages is safe.
    """
    questions: list[dict] = []
    page = 1
    page_size = 100
    last_payload: dict = {}
    while len(questions) < target_n:
        print(f"  fetching questions page {page}...")
        payload = _se_get(
            "/questions",
            {
                "tagged":   tag,
                "sort":     "votes",
                "order":    "desc",
                "page":     page,
                "pagesize": page_size,
            },
        )
        last_payload = payload
        items = payload.get("items", [])
        if not items:
            # Truly out of results.
            print(f"  page {page} returned 0 items — stopping pagination")
            break
        questions.extend(items)
        page += 1
        # Hard safety cap so a buggy API loop can't blow the daily quota.
        if page > 30:
            print("  hit 30-page safety cap — stopping pagination")
            break
    print(f"  fetched {len(questions)} raw questions "
          f"(quota_remaining={last_payload.get('quota_remaining', '?')})")
    return questions[:target_n]


def fetch_so_answers(answer_ids: list[int]) -> dict[int, dict]:
    """Batch-fetch answer bodies by ID. Returns {answer_id: answer_dict}.

    SE API accepts up to 100 IDs per /answers/{ids} call as a semicolon-
    separated string. We chunk accordingly.

    GOTCHA: the default `pagesize` on this endpoint is 30, not 100. Without
    `pagesize=100` you only get back the first 30 of your 100 requested IDs.
    We hit that and it caused ~50% of accepted answers to silently vanish
    in the first run.
    """
    result: dict[int, dict] = {}
    for i in range(0, len(answer_ids), 100):
        chunk = answer_ids[i:i + 100]
        ids_param = ";".join(str(x) for x in chunk)
        print(f"  fetching answers {i}..{i + len(chunk)}")
        payload = _se_get(
            f"/answers/{ids_param}",
            {
                "filter":   "withbody",  # filter MUST include 'body' field
                "pagesize": 100,         # default is 30 — would truncate response
            },
        )
        for ans in payload.get("items", []):
            result[ans["answer_id"]] = ans
    return result


# =============================================================================
# 3. CLEAN — strip HTML to readable plain text
# =============================================================================
def clean_html(html: str) -> str:
    """Convert SO answer HTML to plain text with code blocks preserved.

    SO answer bodies look like:
        <p>You can use <code>docker ps -a</code> to list...</p>
        <pre><code>$ docker ps -a
        ...</code></pre>

    We want:
        You can use `docker ps -a` to list...

        ```
        $ docker ps -a
        ...
        ```

    so the training data reads like a normal markdown answer, not HTML soup.
    BeautifulSoup with a hand-rolled walk over the parse tree gives us
    control over which tags become which whitespace — `get_text()` alone
    collapses code blocks into a single line, which corrupts shell commands.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Convert <pre><code>…</code></pre> to fenced markdown BEFORE we extract
    # text, so newlines inside code blocks survive.
    for pre in soup.find_all("pre"):
        code_text = pre.get_text()
        pre.replace_with(f"\n```\n{code_text.rstrip()}\n```\n")

    # Inline <code> → backticks.
    for code in soup.find_all("code"):
        code.replace_with(f"`{code.get_text()}`")

    # Paragraphs and line breaks become real newlines.
    for tag in soup.find_all(["p", "br", "li"]):
        tag.insert_after("\n")

    text = soup.get_text()
    # Collapse 3+ newlines to 2 (markdown paragraph spacing).
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip()


# =============================================================================
# 4. FILTER — quality cuts before dedup
# =============================================================================
# These thresholds are the easy first-pass cuts. They're cheap (no embedding
# model needed) and we apply them before the more expensive semantic dedup
# step so we embed fewer questions.
MIN_QUESTION_SCORE = 5       # SO community voted up the question at least 5x
MIN_ANSWER_CHARS   = 50      # filter out one-liner answers (no learning signal)
MAX_ANSWER_CHARS   = 4000    # filter out megaposts (likely tutorials, off-topic)
MIN_QUESTION_CHARS = 15      # ditch ultra-terse titles ("docker?")
MAX_QUESTION_CHARS = 300     # filter out questions with the whole bug report in the title


def filter_candidates(
    questions: list[dict],
    answers: dict[int, dict],
) -> list[TrainExample]:
    """Apply quality cuts and build TrainExample rows.

    Each surviving SO question yields exactly one training row (title +
    accepted answer body). We log how many rows fell at each cut so it's
    obvious if a threshold is too strict.

    Also dedupes by so_question_id within the input batch — SO paging can
    return the same question on multiple pages when scores shift between
    requests, and we don't want the model seeing the same Q&A twice.
    """
    rejects = {
        "no_accepted":    0,  # question had no accepted answer
        "missing_answer": 0,  # accepted_answer_id wasn't in the answers dict
        "low_score":      0,  # question vote count below MIN_QUESTION_SCORE
        "q_too_short":    0,
        "q_too_long":     0,
        "a_too_short":    0,
        "a_too_long":     0,
        "duplicate_id":   0,  # same so_question_id appeared earlier in the batch
    }
    out: list[TrainExample] = []
    seen_q_ids: set[int] = set()

    for q in questions:
        if q["question_id"] in seen_q_ids:
            rejects["duplicate_id"] += 1
            continue
        seen_q_ids.add(q["question_id"])
        ans_id = q.get("accepted_answer_id")
        if ans_id is None:
            rejects["no_accepted"] += 1
            continue
        ans = answers.get(ans_id)
        if ans is None:
            rejects["missing_answer"] += 1
            continue
        if q.get("score", 0) < MIN_QUESTION_SCORE:
            rejects["low_score"] += 1
            continue

        # SO question titles are unescaped HTML — &amp; → &, etc.
        title = BeautifulSoup(q["title"], "html.parser").get_text().strip()
        if len(title) < MIN_QUESTION_CHARS:
            rejects["q_too_short"] += 1
            continue
        if len(title) > MAX_QUESTION_CHARS:
            rejects["q_too_long"] += 1
            continue

        answer_text = clean_html(ans.get("body", ""))
        if len(answer_text) < MIN_ANSWER_CHARS:
            rejects["a_too_short"] += 1
            continue
        if len(answer_text) > MAX_ANSWER_CHARS:
            rejects["a_too_long"] += 1
            continue

        out.append(TrainExample(
            question=title,
            answer=answer_text,
            so_question_id=q["question_id"],
            score=q.get("score", 0),
            source_url=q.get("link", f"https://stackoverflow.com/q/{q['question_id']}"),
        ))

    print(f"  filter stats: kept {len(out)}, rejects={rejects}")
    return out


# =============================================================================
# 5. DEDUP against the eval set (the "disjoint rule" enforcer)
# =============================================================================
# Why this matters again, in one sentence: training on a paraphrase of an
# eval question == open-book testing == invalid benchmark.
#
# How it works:
#   - Embed all eval questions with sentence-transformers/all-MiniLM-L6-v2
#     (~80MB, CPU-only, ~1ms per sentence). Each question becomes a 384-d
#     unit vector.
#   - Do the same for each candidate training question.
#   - Compute cosine similarity between every candidate and every eval
#     question. If the candidate's MAX similarity to any eval question
#     exceeds the threshold, drop it.
#
# THRESHOLD = 0.75:
#   In MiniLM space, 0.85+ = near-paraphrase, 0.65-0.85 = same topic
#   different angle, <0.5 = unrelated. 0.75 is the standard "drop
#   paraphrases but keep different-angle questions" cut. Tighter (e.g. 0.85)
#   leaks paraphrases; looser (e.g. 0.65) deletes legitimately different
#   Docker questions.
DEDUP_SIMILARITY_THRESHOLD = 0.75
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def dedup_against_eval(
    candidates: list[TrainExample],
    eval_questions: list[str],
    threshold: float = DEDUP_SIMILARITY_THRESHOLD,
) -> tuple[list[TrainExample], list[tuple[TrainExample, float, str]]]:
    """Drop training candidates that semantically overlap with any eval question.

    Returns (kept_rows, dropped_rows_with_match_info). The second list is for
    spot-checking — we want to eyeball a few "dropped" rows to confirm the
    threshold is sane.
    """
    # Lazy import: this is the first place we need torch loaded, and importing
    # at module top would slow down `import build_train` for callers that
    # only need the SCHEMA (e.g. validation tests).
    from sentence_transformers import SentenceTransformer
    import numpy as np

    print(f"  loading embedding model {EMBED_MODEL_NAME} (one-time)...")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    print(f"  embedding {len(eval_questions)} eval questions...")
    eval_emb = model.encode(
        eval_questions,
        normalize_embeddings=True,   # so cosine = dot product
        show_progress_bar=False,
    )

    print(f"  embedding {len(candidates)} training candidates...")
    cand_texts = [c.question for c in candidates]
    cand_emb = model.encode(
        cand_texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=64,
    )

    # sims[i, j] = cosine similarity between candidate i and eval question j.
    sims = cand_emb @ eval_emb.T          # (n_cand, n_eval)
    max_sim_per_cand = sims.max(axis=1)   # (n_cand,)
    nearest_eval_idx = sims.argmax(axis=1)

    kept: list[TrainExample] = []
    dropped: list[tuple[TrainExample, float, str]] = []
    for i, cand in enumerate(candidates):
        if max_sim_per_cand[i] >= threshold:
            dropped.append((
                cand,
                float(max_sim_per_cand[i]),
                eval_questions[nearest_eval_idx[i]],
            ))
        else:
            kept.append(cand)

    print(f"  dedup: kept {len(kept)}, dropped {len(dropped)} "
          f"(threshold cos≥{threshold})")
    return kept, dropped


# =============================================================================
# 6. WRITE
# =============================================================================
def write_train_jsonl(rows: Iterable[TrainExample], path: Path) -> int:
    """Write training rows to JSONL. Returns the count written.

    Sorted by so_question_id for stable diffs (mirrors save_eval's pattern).
    """
    rows_list = sorted(rows, key=lambda r: r.so_question_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows_list:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    return len(rows_list)


def load_train(path: Path) -> list[TrainExample]:
    """Read train_1k.jsonl back into TrainExample objects (for spot-check / reuse)."""
    out: list[TrainExample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(TrainExample(**json.loads(line)))
    return out
