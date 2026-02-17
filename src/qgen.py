#!/usr/bin/env python3
"""
qgen.py — Generate + verify Turkish "Sözel Mantık" questions from sample_questions.json.

Design goals:
- Minimal surface area (single file)
- Structured Outputs (JSON Schema) for reliable parsing
- Two-step pipeline: generate -> verify (LLM-as-checker)
- Hard budget cap
- Robust against:
  - models that don't support temperature
  - incomplete responses due to max_output_tokens
- Crash-safe output:
  - writes incrementally (after each sample)
  - on any exception, flushes partial results to --out
  - atomic writes to avoid corrupt JSON
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI


# -----------------------------
# Pricing (USD per 1M tokens)
# Keep these in sync with OpenAI pricing docs.
# If a model is missing, we use conservative defaults.
# -----------------------------
MODEL_PRICING_PER_1M = {
    "gpt-4o-mini": {"in": 0.15, "out": 0.60},
    "gpt-4.1-mini": {"in": 0.40, "out": 1.60},
    "gpt-5-mini": {"in": 0.25, "out": 2.00},
    "gpt-5.2": {"in": 1.75, "out": 14.00},
}


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    p = MODEL_PRICING_PER_1M.get(model)
    if not p:
        # Conservative fallback if you change models without updating pricing.
        p = {"in": 1.0, "out": 3.0}
    return (input_tokens * p["in"] + output_tokens * p["out"]) / 1_000_000.0


def get_usage(resp: Any) -> Tuple[int, int]:
    """Returns (input_tokens, output_tokens). Defensive across SDK versions."""
    u = getattr(resp, "usage", None)
    if not u:
        return 0, 0
    in_t = getattr(u, "input_tokens", None)
    out_t = getattr(u, "output_tokens", None)
    if in_t is None:
        in_t = getattr(u, "prompt_tokens", 0)
    if out_t is None:
        out_t = getattr(u, "completion_tokens", 0)
    return int(in_t or 0), int(out_t or 0)


def extract_output_text(resp: Any) -> str:
    """Responses API usually provides resp.output_text. Otherwise scan output items."""
    txt = getattr(resp, "output_text", None)
    if isinstance(txt, str) and txt.strip():
        return txt

    out = getattr(resp, "output", None)
    if isinstance(out, list):
        chunks: List[str] = []
        for item in out:
            if isinstance(item, dict):
                content = item.get("content", [])
                if isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict) and c.get("type") == "output_text":
                            t = c.get("text", "")
                            if isinstance(t, str):
                                chunks.append(t)
        if chunks:
            return "\n".join(chunks)

    raise RuntimeError("Could not extract output text from model response.")


class IncompleteResponseError(RuntimeError):
    pass


def _supports_temperature_error(msg: str) -> bool:
    msg_l = msg.lower()
    return ("unsupported parameter" in msg_l) and ("temperature" in msg_l)


def call_responses_create_with_retry(
    client: OpenAI,
    *,
    model: str,
    system: str,
    user: str,
    json_schema: Dict[str, Any],
    max_output_tokens: int,
    max_output_tokens_cap: int,
    temperature: float,
    retries: int = 5,
) -> Tuple[Dict[str, Any], int, int]:
    """
    Calls Responses API with Structured Outputs.
    - If model rejects `temperature`, retries without it.
    - If response is incomplete due to max_output_tokens, ramps tokens up and retries.
    """
    last_err: Optional[Exception] = None
    backoff = 1.0
    current_max_out = max_output_tokens

    for attempt in range(1, retries + 1):
        try:
            base_kwargs = dict(
                model=model,
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": json_schema.get("name", "schema"),
                        "strict": True,
                        "schema": json_schema["schema"],
                    }
                },
                max_output_tokens=current_max_out,
                store=False,
            )

            # Try with temperature; if not supported, retry without it once.
            try:
                resp = client.responses.create(**base_kwargs, temperature=temperature)
            except Exception as e:
                msg = str(e)
                if _supports_temperature_error(msg):
                    resp = client.responses.create(**base_kwargs)
                else:
                    raise

            status = getattr(resp, "status", None)
            if status == "incomplete":
                raise IncompleteResponseError(
                    f"Model response incomplete (max_output_tokens={current_max_out} reached)."
                )

            raw = extract_output_text(resp)
            parsed = json.loads(raw)

            in_t, out_t = get_usage(resp)
            return parsed, in_t, out_t

        except IncompleteResponseError as e:
            last_err = e
            # ramp output tokens for next attempt
            if current_max_out < max_output_tokens_cap:
                current_max_out = min(int(current_max_out * 1.7) + 256, max_output_tokens_cap)
                continue  # retry immediately
        except Exception as e:
            last_err = e

        if attempt == retries:
            break
        time.sleep(backoff)
        backoff = min(backoff * 2.0, 8.0)

    raise RuntimeError(f"API call failed after {retries} retries: {last_err}")


def write_json_atomic(path: Path, data: Any) -> None:
    """
    Atomic JSON write:
    - write to temp file next to target
    - fsync
    - replace target
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# -----------------------------
# Schemas (Structured Outputs)
# Keep lengths bounded to reduce token usage and incomplete outputs.
# -----------------------------
GEN_SCHEMA = {
    "name": "logic_generation_batch",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "context": {"type": "string", "minLength": 50, "maxLength": 2000},
            "items": {
                "type": "array",
                "minItems": 5,
                "maxItems": 5,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "question": {"type": "string", "minLength": 10, "maxLength": 350},
                        "choices": {
                            "type": "array",
                            "minItems": 5,
                            "maxItems": 5,
                            "items": {"type": "string", "minLength": 3, "maxLength": 180},
                        },
                        "answer": {"type": "string", "enum": ["A", "B", "C", "D", "E"]},
                    },
                    "required": ["question", "choices", "answer"],
                },
            },
        },
        "required": ["context", "items"],
    },
}

VER_SCHEMA = {
    "name": "logic_verification_batch",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "overall_pass": {"type": "boolean"},
            "items": {
                "type": "array",
                "minItems": 5,
                "maxItems": 5,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "pass": {"type": "boolean"},
                        "correct_answer": {"type": "string", "enum": ["A", "B", "C", "D", "E", "UNKNOWN"]},
                        "unique": {"type": "boolean"},
                        "solvable": {"type": "boolean"},
                        "notes": {"type": "string", "maxLength": 220},
                        "confidence": {"type": "number"},
                    },
                    "required": ["pass", "correct_answer", "unique", "solvable", "notes", "confidence"],
                },
            },
        },
        "required": ["overall_pass", "items"],
    },
}


# -----------------------------
# Prompts
# -----------------------------
GEN_SYSTEM = (
    "Sen Türkçe standart sınavlar (ALES, DGS, KPSS) için 'Sözel Mantık' soruları yazan bir uzmansın. "
    "Sadece istenen JSON şemasına uygun çıktı ver. "
    "Gereksiz uzun metin yazma."
)

VER_SYSTEM = (
    "Sen çok titiz bir sınav denetçisisin. "
    "Her sorunun tek doğru şıkkı olup olmadığını ve çözülebilirliğini kontrol edeceksin. "
    "Sadece istenen JSON şemasına uygun çıktı ver. "
    "Notları kısa tut."
)


def build_generator_user_prompt(sample_obj: Dict[str, Any]) -> str:
    sample_text = json.dumps(sample_obj, ensure_ascii=False, indent=2)
    return f"""
Aşağıdaki örnek sadece TARZ içindir. Aynı zorluk hissini koru ama TAMAMEN YENİ bir senaryo üret.

Zorunlular:
- Dil: Türkçe.
- Yeni senaryo (context) örnekteki temayı / isimleri / nesneleri kopyalamayacak.
- Kurallar 5–9 madde olacak. Çelişki olmayacak.
- 5 adet çoktan seçmeli soru üretilecek ve her soruda 5 şık (A–E) olacak.
- Her soruda yalnızca 1 doğru şık olacak.
- Context kısa ve sınav tarzında olsun (gereksiz uzun olmasın).

Soru çeşitleri (aynı context üzerinden):
1) Kesin doğru (hangisi kesin doğrudur?)
2) Kesin yanlış (hangisi kesin yanlıştır?)
3) Olabilir (hangisi mümkün olabilir?)
4) Koşullu çıkarım (Eğer X ise...)
5) Daha zor/karma çıkarım

ÖRNEK (sadece stil referansı):
{sample_text}

Şimdi yeni bir context ve 5 soru üret.
""".strip()


def build_verifier_user_prompt(context: str, items: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("CONTEXT:")
    lines.append(context.strip())
    lines.append("")
    lines.append("SORULAR:")

    for i, it in enumerate(items, start=1):
        lines.append(f"\nSoru {i}: {it['question'].strip()}")
        for c in it["choices"]:
            lines.append(c.strip())

    lines.append(
        "\nTalimat: Her soruyu çöz; tek doğru şık olduğundan emin ol; belirsizlik/çelişki varsa FAIL. "
        "Cevap emin değilse UNKNOWN yaz."
    )
    return "\n".join(lines)


# -----------------------------
# Core pipeline
# -----------------------------
@dataclass
class Budget:
    max_usd: float
    spent_usd: float = 0.0

    def can_spend_more(self) -> bool:
        return self.spent_usd < self.max_usd


def validate_generated_shape(gen: Dict[str, Any]) -> None:
    if not isinstance(gen, dict):
        raise ValueError("Generator output is not an object.")
    if not isinstance(gen.get("context"), str) or not gen["context"].strip():
        raise ValueError("Missing/empty context.")
    items = gen.get("items")
    if not isinstance(items, list) or len(items) != 5:
        raise ValueError("Generator must return exactly 5 items.")
    for it in items:
        if not isinstance(it, dict):
            raise ValueError("Item is not an object.")
        if not isinstance(it.get("question"), str) or not it["question"].strip():
            raise ValueError("Missing/empty question.")
        choices = it.get("choices")
        if not isinstance(choices, list) or len(choices) != 5 or not all(isinstance(c, str) for c in choices):
            raise ValueError("Each item must have 5 string choices.")
        ans = it.get("answer")
        if ans not in ("A", "B", "C", "D", "E"):
            raise ValueError("Each item must include answer A–E.")


def generate_and_verify_one_batch(
    client: OpenAI,
    *,
    sample_obj: Dict[str, Any],
    gen_model: str,
    ver_model: str,
    gen_max_output_tokens: int,
    gen_max_output_tokens_cap: int,
    ver_max_output_tokens: int,
    ver_max_output_tokens_cap: int,
    budget: Budget,
    gen_temperature: float,
    ver_temperature: float,
    min_confidence: float,
) -> Tuple[List[Dict[str, Any]], float]:
    batch_cost = 0.0

    # 1) Generate
    gen_prompt = build_generator_user_prompt(sample_obj)
    gen_json, in_t, out_t = call_responses_create_with_retry(
        client,
        model=gen_model,
        system=GEN_SYSTEM,
        user=gen_prompt,
        json_schema=GEN_SCHEMA,
        max_output_tokens=gen_max_output_tokens,
        max_output_tokens_cap=gen_max_output_tokens_cap,
        temperature=gen_temperature,
        retries=5,
    )
    validate_generated_shape(gen_json)
    c = estimate_cost_usd(gen_model, in_t, out_t)
    batch_cost += c
    budget.spent_usd += c

    if not budget.can_spend_more():
        return [], batch_cost

    context = gen_json["context"]
    gen_items = gen_json["items"]

    # 2) Verify (do not provide generator answers)
    ver_prompt = build_verifier_user_prompt(context, gen_items)
    ver_json, vin_t, vout_t = call_responses_create_with_retry(
        client,
        model=ver_model,
        system=VER_SYSTEM,
        user=ver_prompt,
        json_schema=VER_SCHEMA,
        max_output_tokens=ver_max_output_tokens,
        max_output_tokens_cap=ver_max_output_tokens_cap,
        temperature=ver_temperature,
        retries=5,
    )
    vc = estimate_cost_usd(ver_model, vin_t, vout_t)
    batch_cost += vc
    budget.spent_usd += vc

    passed: List[Dict[str, Any]] = []
    ver_items = ver_json.get("items", [])
    if not isinstance(ver_items, list) or len(ver_items) != 5:
        return [], batch_cost

    for g_it, v_it in zip(gen_items, ver_items):
        ok = (
            isinstance(v_it, dict)
            and v_it.get("pass") is True
            and v_it.get("unique") is True
            and v_it.get("solvable") is True
            and v_it.get("correct_answer") in ("A", "B", "C", "D", "E")
            and float(v_it.get("confidence", 0.0)) >= min_confidence
        )
        if ok:
            passed.append(
                {
                    "context": context,
                    "question": g_it["question"],
                    "choices": g_it["choices"],
                    "answer": v_it["correct_answer"],  # verifier decides
                }
            )

    return passed, batch_cost


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate + verify Turkish verbal logic questions.")
    ap.add_argument("--samples", required=True, help="Path to sample_questions.json")
    ap.add_argument("--out", required=True, help="Output JSON file path")
    ap.add_argument("--limit", type=int, default=10, help="How many sample questions to process")
    ap.add_argument("--max-usd", type=float, default=5.0, help="Hard budget cap in USD")
    ap.add_argument("--max-attempts", type=int, default=3, help="Max generate/verify attempts per sample question")

    ap.add_argument("--gen-model", default=os.getenv("GEN_MODEL", "gpt-4o-mini"))
    ap.add_argument("--ver-model", default=os.getenv("VER_MODEL", "gpt-4.1-mini"))

    # Higher defaults to avoid incomplete responses
    ap.add_argument("--gen-max-output-tokens", type=int, default=3200)
    ap.add_argument("--ver-max-output-tokens", type=int, default=2200)

    # Caps used only if response becomes incomplete
    ap.add_argument("--gen-max-output-tokens-cap", type=int, default=6000)
    ap.add_argument("--ver-max-output-tokens-cap", type=int, default=4000)

    ap.add_argument("--gen-temp", type=float, default=0.5)
    ap.add_argument("--ver-temp", type=float, default=0.0)

    ap.add_argument("--min-confidence", type=float, default=0.65)
    args = ap.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set.")

    samples_path = Path(args.samples)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    samples = json.loads(samples_path.read_text(encoding="utf-8"))
    if not isinstance(samples, list):
        raise SystemExit("samples file must be a JSON array.")
    samples = samples[: max(args.limit, 0)]

    client = OpenAI(api_key=api_key)
    budget = Budget(max_usd=args.max_usd)

    all_outputs: List[Dict[str, Any]] = []

    def flush_partial() -> None:
        # Always keep the output file valid JSON
        write_json_atomic(out_path, all_outputs)

    try:
        for idx, sample in enumerate(samples, start=1):
            if not budget.can_spend_more():
                break

            source_id = str(sample.get("id", f"sample_{idx}"))
            collected: List[Dict[str, Any]] = []

            for _attempt in range(1, args.max_attempts + 1):
                if not budget.can_spend_more() or len(collected) >= 5:
                    break

                passed, _ = generate_and_verify_one_batch(
                    client,
                    sample_obj=sample,
                    gen_model=args.gen_model,
                    ver_model=args.ver_model,
                    gen_max_output_tokens=args.gen_max_output_tokens,
                    gen_max_output_tokens_cap=args.gen_max_output_tokens_cap,
                    ver_max_output_tokens=args.ver_max_output_tokens,
                    ver_max_output_tokens_cap=args.ver_max_output_tokens_cap,
                    budget=budget,
                    gen_temperature=args.gen_temp,
                    ver_temperature=args.ver_temp,
                    min_confidence=args.min_confidence,
                )

                for it in passed:
                    if len(collected) >= 5:
                        break
                    new_id = f"{source_id}__syn_{len(collected)+1}"
                    collected.append(
                        {
                            "id": new_id,
                            "context": it["context"],
                            "question": it["question"],
                            "choices": it["choices"],
                            "answer": it["answer"],
                        }
                    )

            all_outputs.extend(collected)

            # Flush after each sample so partial progress is always saved.
            flush_partial()

            print(
                f"[{idx}/{len(samples)}] {source_id}: kept={len(collected)}/5 | "
                f"spent=${budget.spent_usd:.2f}/${budget.max_usd:.2f}"
            )

        # Final flush (already flushed each loop, but keep it explicit)
        flush_partial()
        print(f"\nWrote {len(all_outputs)} questions to: {out_path}")
        print(f"Estimated spend: ${budget.spent_usd:.2f} (cap: ${budget.max_usd:.2f})")

    except KeyboardInterrupt:
        # Ctrl+C => still write partial results
        print("\nInterrupted. Writing partial output...", file=sys.stderr)
        try:
            flush_partial()
        except Exception as e:
            print(f"Failed to write partial output: {e}", file=sys.stderr)
        raise

    except Exception as e:
        # Any crash => still write partial results
        print(f"\nError: {e}\nWriting partial output...", file=sys.stderr)
        try:
            flush_partial()
        except Exception as ee:
            print(f"Failed to write partial output: {ee}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
