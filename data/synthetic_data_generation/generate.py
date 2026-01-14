#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate.py — Turkish logic benchmark expander (LLM-driven) + local fallback

(PG, no heresy):
- If you ask an LLM for "similar questions", it will happily give you 10 clones wearing different moustaches.
- So we enforce STRUCTURE, DEDUPE, and "DON'T REPEAT YOURSELF" like we're raising a stubborn compiler.
- You're building a benchmark. Benchmarks need variety, or you’re just measuring memorization in funny hats.

What this file does (minimal, but useful):
1) Reads seed exam questions from a text file or JSONL.
2) Calls OpenAI (cheapest default model: gpt-5-nano) to generate new Turkish questions:
   - varied themes/scenarios
   - varied question templates
   - varied difficulty (1..5)
3) Writes JSONL output, de-duplicated (doesn't keep outputting the same wrapper forever).
4) Works WITHOUT an API key too (local generator), but LLM mode is what you want.

IMPORTANT REALITY CHECK:
- If you want a TRUE benchmark (auto-scored), you need correct answers.
- This script can request the LLM to include answers, but they can be wrong.
- So we store `answer_source="llm"` and `quality_flags` so you can filter/verify later.
- If you later add a solver/validator, you can flip answers to verified.

Usage examples:
  # Expand dataset using OpenAI (best for themes/templates diversity)
  python generate.py --mode llm_expand --seed-file seeds.txt --out out.jsonl --n 200

  # Same, but sample from a JSONL dataset
  python generate.py --mode llm_expand --seed-file seeds.jsonl --out out.jsonl --n 200

  # Keep a rolling dedupe index across runs
  python generate.py --mode llm_expand --seed-file seeds.txt --out out.jsonl --n 200 --dedupe-index out.dedupe.txt

  # Local fallback (no API key): still varied THEMES (not only ships)
  python generate.py --mode local --out out.jsonl --n 50

Seed formats:
- .txt: one question block separated by blank lines (two newlines).
- .jsonl: expects fields like: stem, question, choices (list or string), answer (optional).
  We’ll try best-effort extraction.

Output JSONL schema (per question row):
{
  id, locale, split, theme, difficulty,
  stem, question, choices[5], answer, answer_source,
  qtype, seed_hash, generation_meta, quality_flags, raw
}

"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Optional: load .env if you have it (won't crash if missing)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# Optional OpenAI SDK
try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore


# ---------------------------
# Small utilities
# ---------------------------

LETTERS = ["A", "B", "C", "D", "E"]

def sha1_hex(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def normalize_text(s: str) -> str:
    # Normalize for dedupe: lowercase, trim, collapse whitespace, kill punctuation-ish noise.
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[“”\"'’`]", "", s)
    s = re.sub(r"\s*([.,;:!?])\s*", r"\1", s)
    return s

def fingerprint(stem: str, question: str, choices: List[str]) -> str:
    key = normalize_text(stem) + "||" + normalize_text(question) + "||" + "||".join(normalize_text(c) for c in choices)
    return sha1_hex(key)

def stable_split(example_id: str, ratios=(0.8, 0.1, 0.1)) -> str:
    # Deterministic split so reruns don't shuffle the universe.
    h = int(sha1_hex(example_id)[:8], 16) % 10_000
    x = h / 10_000.0
    a, b, _c = ratios
    if x < a:
        return "train"
    if x < a + b:
        return "dev"
    return "test"

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def load_dedupe_index(path: Optional[str]) -> set:
    if not path or not os.path.exists(path):
        return set()
    out = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            x = line.strip()
            if x:
                out.add(x)
    return out

def append_dedupe_index(path: Optional[str], fps: List[str]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for x in fps:
            f.write(x + "\n")


# ---------------------------
# Seed loading
# ---------------------------

def load_seeds(seed_file: str, max_seeds: int = 5000) -> List[str]:
    """
    Returns seed examples as plain text blocks.
    """
    if not os.path.exists(seed_file):
        raise FileNotFoundError(seed_file)

    if seed_file.endswith(".jsonl"):
        seeds: List[str] = []
        with open(seed_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                stem = str(obj.get("stem") or obj.get("stem_tr") or obj.get("context") or "")
                q = str(obj.get("question") or obj.get("question_tr") or "")
                choices = obj.get("choices") or obj.get("choices_tr") or obj.get("options") or []
                if isinstance(choices, str):
                    # could be "A) ...\nB) ..."
                    c_list = [x.strip() for x in choices.splitlines() if x.strip()]
                    choices_list = c_list
                elif isinstance(choices, list):
                    choices_list = [str(x) for x in choices]
                else:
                    choices_list = []
                if stem or q:
                    block = stem.strip()
                    if q.strip():
                        block += ("\n\n" if block else "") + q.strip()
                    if choices_list:
                        block += "\n" + "\n".join(choices_list[:10])
                    seeds.append(block.strip())
                if len(seeds) >= max_seeds:
                    break
        if not seeds:
            raise ValueError("No seeds loaded from JSONL. Expected fields like stem/question/choices.")
        return seeds

    # .txt: blocks separated by blank lines
    raw = read_text(seed_file).strip()
    blocks = [b.strip() for b in re.split(r"\n\s*\n\s*\n+", raw) if b.strip()]
    if not blocks:
        # fallback: split by a single blank line
        blocks = [b.strip() for b in re.split(r"\n\s*\n+", raw) if b.strip()]
    if not blocks:
        raise ValueError("No seed blocks found in text file.")
    return blocks[:max_seeds]


# ---------------------------
# Themes & qtypes
# ---------------------------

THEMES = [
    "ships_port",
    "exam_entry",
    "project_presentations",
    "hospital_appointments",
    "cargo_deliveries",
    "call_center",
    "flight_departures",
    "race_finish",
    "lab_experiments",
    "conference_talks",
    "kitchen_orders",
    "workshop_schedule",
]

QTYPES = [
    "could_be_pos",
    "definitely_true",
    "definitely_false",
    "pair_not_adjacent",
    "additional_info_unique",
    "earliest_possible",
    "latest_possible",
    "contradiction_check",
]

# ---------------------------
# OpenAI Structured Output schema
# ---------------------------

# Keep schema simple. Fancy JSON Schema features can be flaky across tooling.
GEN_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "generated": {
            "type": "array",
            "minItems": 1,
            "maxItems": 20,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "theme": {"type": "string"},
                    "difficulty": {"type": "integer", "minimum": 1, "maximum": 5},
                    "title_tr": {"type": "string"},
                    "stem_tr": {"type": "string"},
                    "questions": {
                        "type": "array",
                        "minItems": 3,
                        "maxItems": 6,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "qtype": {"type": "string"},
                                "question_tr": {"type": "string"},
                                "choices_tr": {
                                    "type": "array",
                                    "minItems": 5,
                                    "maxItems": 5,
                                    "items": {"type": "string"},
                                },
                                # We ask for an answer letter so it is benchmark-ready.
                                # But it might be wrong; we label answer_source="llm".
                                "answer": {"type": "string"},
                                "explanation_tr": {"type": "string"},
                            },
                            "required": ["qtype", "question_tr", "choices_tr", "answer"],
                        },
                    },
                    # Optional: store a formal spec for later verification/solving.
                    "items": {"type": "array", "items": {"type": "string"}},
                    "constraints_formal": {"type": "array", "items": {"type": "object"}},
                    "notes": {"type": "string"},
                },
                "required": ["theme", "difficulty", "title_tr", "stem_tr", "questions"],
            },
        }
    },
    "required": ["generated"],
    "additionalProperties": False,
}

def coerce_answer_letter(x: str) -> str:
    x = x.strip().upper()
    # Accept "A) ..." too.
    if x and x[0] in LETTERS:
        return x[0]
    return "A"

def clean_choices(choices: List[str]) -> List[str]:
    # Normalize to "A) ..." format for output consistency.
    out = []
    for i, c in enumerate(choices[:5]):
        c = c.strip()
        # Strip any existing letter prefix
        c = re.sub(r"^[A-Ea-e]\)\s*", "", c)
        out.append(f"{LETTERS[i]}) {c}")
    # Pad if model returns fewer than 5 (shouldn't happen under schema, but…)
    while len(out) < 5:
        out.append(f"{LETTERS[len(out)]}) [EKSIK]")
    return out[:5]


# ---------------------------
# OpenAI call
# ---------------------------

def require_openai() -> Any:
    if OpenAI is None:
        raise RuntimeError("openai SDK not installed. pip install openai")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set.")
    return OpenAI()

def build_prompt(seed_example: str,
                 n_generate: int,
                 recent_themes: List[str],
                 avoid_hashes: List[str]) -> List[Dict[str, str]]:
    """
    We force diversity explicitly:
    - different theme than recent themes when possible
    - mix qtypes
    - vary difficulty
    - avoid paraphrasing the seed too closely
    """
    # Keep avoid info short to not waste tokens (cheap model, remember).
    avoid_note = ""
    if recent_themes:
        avoid_note += f"\n- Avoid these recent themes if possible: {', '.join(recent_themes[:8])}."
    if avoid_hashes:
        avoid_note += "\n- Do NOT produce near-duplicates of previously generated items."

    system = (
        "You are an expert Turkish exam question writer.\n"
        "You generate LOGIC / ORDERING style multiple-choice questions.\n"
        "You MUST output ONLY valid JSON that matches the provided schema.\n"
        "You MUST generate diverse themes, diverse question templates, and diverse difficulty.\n"
        "Do not copy the seed; imitate its reasoning style.\n"
        "Do not include copyrighted exam identifiers or year/session labels.\n"
    )

    user = (
        f"SEED EXAMPLE (for style and reasoning type only):\n{seed_example}\n\n"
        f"Task: Generate {n_generate} NEW Turkish multiple-choice logic items inspired by the seed.\n"
        f"Requirements:\n"
        f"- Pick theme from this list (use variety): {THEMES}\n"
        f"- Use a mix of qtypes from this list (use variety): {QTYPES}\n"
        f"- Each generated puzzle must include 3-6 questions.\n"
        f"- Each question must have EXACTLY 5 options.\n"
        f"- Provide answer as a letter A-E (even if you are not 100% sure).\n"
        f"- Vary difficulty across 1..5; harder means more constraints / trickier inference.\n"
        f"- Use different entity sets (names/teams/places/appointments/etc.) across outputs.\n"
        f"- Do NOT keep repeating the same story wrapper.\n"
        f"{avoid_note}\n\n"
        f"Optional but recommended:\n"
        f"- Include items[] and constraints_formal[] if you can. Keep constraints consistent with stem.\n"
        f"- constraints_formal can be a simple list like:\n"
        f"  {{\"type\":\"fixed_pos\",\"item\":\"X\",\"pos\":6}}, {{\"type\":\"before\",\"a\":\"X\",\"b\":\"Y\"}}, "
        f"{{\"type\":\"adjacent\",\"a\":\"X\",\"b\":\"Y\"}}, {{\"type\":\"immediate_before\",\"a\":\"X\",\"b\":\"Y\"}}.\n"
        f"Don't overcomplicate the formal spec. Consistency > cleverness.\n"
    )
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]

def openai_generate(seed_example: str,
                    n_generate: int,
                    model: str,
                    temperature: float,
                    recent_themes: List[str],
                    avoid_hashes: List[str],
                    store: bool = False) -> Dict[str, Any]:
    client = require_openai()
    messages = build_prompt(seed_example, n_generate, recent_themes, avoid_hashes)

    # Responses API structured outputs.
    # In Responses, structured outputs are specified via text.format json_schema.
    # Keep it strict so JSON is machine-readable.
    resp = client.responses.create(
        model=model,
        input=messages,
        temperature=temperature,
        text={
            "format": {
                "type": "json_schema",
                "name": "turkish_logic_benchmark_batch",
                "strict": True,
                "schema": GEN_SCHEMA,
            }
        },
        # Responses are stored by default in many setups; you can disable.
        store=store,
    )

    # SDKs vary: try output_text first, then fall back.
    out_text = getattr(resp, "output_text", None)
    if not out_text:
        out_text = extract_output_text(resp)
    return json.loads(out_text)

def extract_output_text(resp: Any) -> str:
    # Best-effort extraction from Responses structure
    if hasattr(resp, "output"):
        for item in resp.output:
            if getattr(item, "type", None) == "message":
                content = getattr(item, "content", None) or []
                for c in content:
                    if getattr(c, "type", None) == "output_text":
                        return getattr(c, "text", "")
    if hasattr(resp, "model_dump"):
        dumped = resp.model_dump()
        for item in dumped.get("output", []):
            if item.get("type") == "message":
                for c in item.get("content", []):
                    if c.get("type") == "output_text":
                        return c.get("text", "")
    raise RuntimeError("Could not extract output_text from Responses response.")


# ---------------------------
# Local fallback generator (NO API KEY)
# ---------------------------

LOCAL_THEME_SNIPPETS = {
    "ships_port": ("Bir limana", "yanaşmıştır"),
    "exam_entry": ("Bir sınav salonuna", "girmiştir"),
    "project_presentations": ("Bir projede", "sunum yapmıştır"),
    "hospital_appointments": ("Bir hastanede", "muayeneye girmiştir"),
    "cargo_deliveries": ("Bir depoya", "teslimat yapmıştır"),
    "call_center": ("Bir çağrı merkezinde", "görüşme yapmıştır"),
    "flight_departures": ("Bir havalimanında", "kalkış yapmıştır"),
    "race_finish": ("Bir yarışta", "finişi görmüştür"),
    "lab_experiments": ("Bir laboratuvarda", "deney yapmıştır"),
    "conference_talks": ("Bir konferansta", "konuşma yapmıştır"),
    "kitchen_orders": ("Bir restoranda", "sipariş hazırlamıştır"),
    "workshop_schedule": ("Bir atölyede", "oturuma katılmıştır"),
}

def local_generate_one(rng: random.Random) -> Dict[str, Any]:
    theme = rng.choice(THEMES)
    opener, verb = LOCAL_THEME_SNIPPETS.get(theme, ("Bir etkinlikte", "yer almıştır"))
    difficulty = rng.randint(1, 5)

    # Super lightweight: just varied wrapper and generic question.
    # This is NOT meant to be perfect logic; it’s an offline fallback.
    entities = [f"Kişi {i}" for i in range(1, 10)]
    rng.shuffle(entities)

    stem = (
        f"{opener}; {', '.join(entities[:9])} farklı zamanlarda {verb}.\n"
        f"Aşağıdaki ifadelerden bazıları verilmiştir:\n"
        f"- 6. sırada {entities[0]} vardır.\n"
        f"- {entities[1]} ile {entities[2]} art arda olmuştur.\n"
        f"- {entities[3]}, {entities[4]}'ten önce gelmiştir.\n"
        f"- {entities[5]}'in hemen ardından {entities[6]} gelmiştir.\n"
        f"(Not: Bu yerel mod sadece çeşitlilik içindir; gerçek benchmark için LLM modunu kullan.)"
    )
    q = {
        "qtype": "could_be_pos",
        "question_tr": "Aşağıdakilerden hangisi 2. sırada olabilir?",
        "choices_tr": [f"{c}" for c in rng.sample(entities[:9], 5)],
        "answer": rng.choice(LETTERS),
        "explanation_tr": "",
    }
    return {
        "theme": theme,
        "difficulty": difficulty,
        "title_tr": "Sıralama problemi",
        "stem_tr": stem,
        "questions": [q, q, q],  # dumb filler; local is not the real path
        "notes": "local_fallback",
    }


# ---------------------------
# Main pipeline
# ---------------------------

def postprocess_and_flatten(batch_obj: Dict[str, Any],
                            seed_hash: str,
                            dedupe_set: set,
                            max_per_puzzle: int,
                            run_meta: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    """
    Takes {generated:[...]} and returns flattened question rows, plus fingerprints, plus themes list.
    """
    rows: List[Dict[str, Any]] = []
    new_fps: List[str] = []
    themes_used: List[str] = []

    generated = batch_obj.get("generated", [])
    for pi, puzzle in enumerate(generated):
        theme = str(puzzle.get("theme", "unknown"))
        difficulty = int(puzzle.get("difficulty", 3))
        title_tr = str(puzzle.get("title_tr", "")).strip()
        stem_tr = str(puzzle.get("stem_tr", "")).strip()
        questions = puzzle.get("questions", [])[:max_per_puzzle]

        themes_used.append(theme)

        # sanity
        if not stem_tr or not questions:
            continue

        for qi, q in enumerate(questions):
            qtype = str(q.get("qtype", "unknown"))
            question_tr = str(q.get("question_tr", "")).strip()
            choices_raw = q.get("choices_tr", [])
            if not isinstance(choices_raw, list) or len(choices_raw) != 5:
                continue

            choices = clean_choices([str(x) for x in choices_raw])
            ans = coerce_answer_letter(str(q.get("answer", "A")))
            explanation = str(q.get("explanation_tr", "")).strip()

            fp = fingerprint(stem_tr, question_tr, choices)
            if fp in dedupe_set:
                continue
            dedupe_set.add(fp)
            new_fps.append(fp)

            ex_id = sha1_hex(f"{seed_hash}::{theme}::{difficulty}::{fp}")[:16]
            split = stable_split(ex_id)

            row = {
                "id": f"tr_logic_{ex_id}",
                "locale": "tr",
                "split": split,
                "theme": theme,
                "difficulty": difficulty,
                "title": title_tr,
                "stem": stem_tr,
                "question": question_tr,
                "choices": choices,
                "answer": ans,
                "answer_source": "llm",
                "qtype": qtype,
                "seed_hash": seed_hash,
                "generation_meta": run_meta,
                "quality_flags": {
                    "unverified_answer": True,
                    "schema_strict": True,
                    "needs_solver_validation": bool(puzzle.get("constraints_formal")),
                    "qtype_known": qtype in QTYPES,
                    "theme_known": theme in THEMES,
                },
                "raw": {
                    # Keep extra optional fields for later validation
                    "items": puzzle.get("items"),
                    "constraints_formal": puzzle.get("constraints_formal"),
                    "notes": puzzle.get("notes", ""),
                    "explanation_tr": explanation,
                }
            }
            rows.append(row)

    return rows, new_fps, themes_used


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["llm_expand", "local"], default="llm_expand",
                    help="llm_expand uses OpenAI; local is fallback without API.")
    ap.add_argument("--seed-file", type=str, default="",
                    help="Path to seeds (.txt blocks or .jsonl dataset). Required for llm_expand.")
    ap.add_argument("--out", type=str, default="out.jsonl", help="Output JSONL path (appended).")
    ap.add_argument("--dedupe-index", type=str, default="out.dedupe.txt",
                    help="Text file storing fingerprints across runs. Empty disables.")
    ap.add_argument("--n", type=int, default=100, help="Number of question rows to produce (best-effort).")
    ap.add_argument("--seed", type=int, default=1337, help="RNG seed for sampling seeds.")
    ap.add_argument("--batch-size", type=int, default=4, help="How many puzzles to request per OpenAI call (1..20).")
    ap.add_argument("--max-questions-per-puzzle", type=int, default=6, help="Limit flattening per puzzle.")
    ap.add_argument("--model", type=str, default="gpt-5-nano", help="OpenAI model. Default is the cheapest.")
    ap.add_argument("--temperature", type=float, default=0.4, help="LLM temperature (0.2-0.7 typical).")
    ap.add_argument("--store", action="store_true", help="Allow OpenAI to store Responses (default off).")
    ap.add_argument("--dry-run", action="store_true", help="Generate but do not write output.")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    t0 = time.time()

    dedupe_path = args.dedupe_index.strip() if args.dedupe_index.strip() else None
    dedupe_set = load_dedupe_index(dedupe_path)

    seeds: List[str] = []
    if args.mode == "llm_expand":
        if not args.seed_file:
            print("[fatal] --seed-file is required in llm_expand mode.", file=sys.stderr)
            return 2
        seeds = load_seeds(args.seed_file)
        if not seeds:
            print("[fatal] No seeds loaded.", file=sys.stderr)
            return 2
        sys.stderr.write(f"[info] loaded {len(seeds)} seeds from {args.seed_file}\n")

    total_rows: List[Dict[str, Any]] = []
    all_new_fps: List[str] = []
    recent_themes: List[str] = []

    # We generate until we reach n rows or we hit a safety cap on attempts.
    attempts = 0
    max_attempts = max(10, args.n // 2 + 20)

    while len(total_rows) < args.n and attempts < max_attempts:
        attempts += 1

        if args.mode == "local":
            # Local fallback batch
            batch_obj = {"generated": [local_generate_one(rng) for _ in range(max(1, args.batch_size))]}
            seed_hash = sha1_hex("local")[:12]
            run_meta = {"mode": "local", "seed": args.seed}
        else:
            seed_example = rng.choice(seeds)
            seed_hash = sha1_hex(seed_example)[:12]

            # Avoid duplicates: pass only *tiny* hints to save tokens.
            avoid_hashes = []  # keep empty (fingerprints are not meaningful to the model)
            run_meta = {
                "mode": "llm_expand",
                "model": args.model,
                "temperature": args.temperature,
                "seed": args.seed,
                "batch_size": args.batch_size,
                "seed_file": os.path.basename(args.seed_file),
            }

            try:
                batch_obj = openai_generate(
                    seed_example=seed_example,
                    n_generate=max(1, min(20, args.batch_size)),
                    model=args.model,
                    temperature=args.temperature,
                    recent_themes=recent_themes[-8:],
                    avoid_hashes=avoid_hashes,
                    store=bool(args.store),
                )
            except Exception as e:
                sys.stderr.write(f"[warn] OpenAI generation failed: {e}\n")
                continue

        rows, new_fps, themes_used = postprocess_and_flatten(
            batch_obj=batch_obj,
            seed_hash=seed_hash,
            dedupe_set=dedupe_set,
            max_per_puzzle=args.max_questions_per_puzzle,
            run_meta=run_meta,
        )

        if themes_used:
            recent_themes.extend(themes_used)

        if not rows:
            continue

        total_rows.extend(rows)
        all_new_fps.extend(new_fps)

        sys.stderr.write(f"[info] attempt={attempts} rows={len(total_rows)}/{args.n} dedupe_size={len(dedupe_set)}\n")

    total_rows = total_rows[:args.n]

    if args.dry_run:
        # Print a sample, don't write.
        sample = total_rows[:2]
        print(json.dumps(sample, ensure_ascii=False, indent=2))
        sys.stderr.write("[done] dry-run complete.\n")
        return 0

    if total_rows:
        write_jsonl(args.out, total_rows)
        append_dedupe_index(dedupe_path, all_new_fps)
        dt = time.time() - t0
        sys.stderr.write(f"[done] wrote {len(total_rows)} rows to {args.out} in {dt:.2f}s\n")
    else:
        sys.stderr.write("[done] no rows produced.\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
