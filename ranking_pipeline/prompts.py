"""Prompt construction and response parsing for the two LLM roles:

  Ranking  -- order the retrieved ESCO candidates by how central they are to
              the sentence (build_rank_messages / parse_ranked_indices).
  Gate audit -- decide whether a sentence in the gate's uncertain band is
              truly skill-relevant (build_gate_audit_messages /
              parse_gate_decision).

IMPORTANT: the prompt strings below are part of the on-disk LLM cache key
(via the prompt-version strings in the entry script). Editing a prompt
without bumping the corresponding version string will silently mix old and
new responses in the cache.
"""

import json
import re

# ---------------------------------------------------------------------------
# Ranking prompt
# ---------------------------------------------------------------------------

RANK_SYSTEM_PROMPT = (
    "You identify which skills a job-description sentence expresses, AND rank "
    "them by how central they are to the sentence. You are given one sentence "
    "and a numbered list of candidate ESCO skills retrieved for it. Return the "
    "numbers of the candidates that the sentence genuinely expresses or "
    "requires, ordered from most to least central -- the skill the sentence is "
    "most clearly about first. Select a candidate only if the sentence clearly "
    "supports it; do not guess. If the sentence expresses none of the "
    "candidates, or expresses no skill at all, return an empty list. "
    'Respond with ONLY a JSON object of the form {"ranked": [<numbers>]} '
    "and nothing else, with the numbers ordered most relevant first."
)


def build_rank_user_prompt(sentence: str, candidates: list[str]) -> str:
    numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(candidates))
    return (
        f"Sentence:\n{sentence}\n\n"
        f"Candidate skills:\n{numbered}\n\n"
        'Return the relevant candidate numbers, most relevant first, as JSON: '
        '{"ranked": [...]}. If none of the candidates apply, return {"ranked": []}.'
    )


def build_rank_messages(
    sentence: str,
    candidates: list[str],
    system_prompt: str,
    few_shot: list[dict] | None = None,
) -> list[dict]:
    """Chat messages for one ranking call: system prompt, optional few-shot
    demonstrations as full user/assistant turns, then the live sentence."""
    msgs: list[dict] = [{"role": "system", "content": system_prompt}]
    for ex in few_shot or []:
        msgs.append({"role": "user",
                     "content": build_rank_user_prompt(ex["sentence"], ex["candidates"])})
        msgs.append({"role": "assistant",
                     "content": json.dumps({"ranked": ex["ranked"]})})
    msgs.append({"role": "user", "content": build_rank_user_prompt(sentence, candidates)})
    return msgs


def parse_ranked_indices(text: str, n_candidates: int) -> list[int]:
    """Extract 0-based candidate indices from the model's JSON reply, in the
    order given (most relevant first). Robust to code fences and trailing
    prose; out-of-range / non-int / duplicate entries are dropped.
    """
    if not text:
        return []
    # Drop any Qwen <think>...</think> reasoning that leaks into content.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    obj = None
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group())
            except json.JSONDecodeError:
                obj = None
    if not isinstance(obj, dict):
        return []
    out: list[int] = []
    seen: set[int] = set()
    for x in obj.get("ranked", []) or []:
        try:
            i = int(x) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= i < n_candidates and i not in seen:
            seen.add(i)
            out.append(i)
    return out


# ---------------------------------------------------------------------------
# Gate-audit prompt (LLM cascade refinement of the binary relevance gate)
# ---------------------------------------------------------------------------

GATE_AUDIT_PROMPT_VERSION = "gate-audit-v1"

GATE_AUDIT_SYSTEM_PROMPT = """You are an AI data extractor for an HR platform.
Your task is to determine if a sentence contains an EXPLICIT, EXTRACTABLE SKILL or QUALIFICATION.
Most of the sentences you will see were considered relevant by the base model; your task is to audit them and determine if they are truly relevant.

RULES:

RELEVANT: The sentence contains:

Concrete hard or soft skills (e.g., Python, data analysis).

Core professional competencies (e.g., "Customer focus").

Specific experience requirements (e.g., "cooking experience").

Domain-specific tasks that inherently require technical or professional expertise (e.g., "preparing cold food items", "managing customer contracts").

NON-RELEVANT: The sentence describes:

Vague/general job duties (e.g., "responsible for your shift", "host clients").

Work conditions, company perks, marketing fluff, or generic behavioral expectations (e.g., "willing to live onboard", "desire for a career").

You must briefly explain your reasoning, then end your response with exactly: "FINAL DECISION: RELEVANT" or "FINAL DECISION: NON-RELEVANT".

--- EXAMPLES ---

--- EXAMPLES OF FALSE POSITIVES (Reject these) ---

Text: "High School Diploma required"
Reasoning: This states a baseline educational background requirement. While a formal credential, it does not represent a specific, extractable hard or soft professional skill.
FINAL DECISION: NON-RELEVANT

Text: "A Bachelor's degree is required and a Master's degree is preferred."
Reasoning: This outlines general educational degrees. General education requirements are background prerequisites, not explicit, mappable domain skills or targeted competencies.
FINAL DECISION: NON-RELEVANT

Text: "Public Housing experience A MUST"
Reasoning: This asks for general background experience within a broad sector (Public Housing). Because it describes a working environment rather than naming a specific, extractable technical or soft skill, it should be excluded.
FINAL DECISION: NON-RELEVANT

--- EXAMPLES OF TRUE POSITIVES (Keep these!) ---

Text: "+ year(s) of medical billing experience."
Reasoning: "Medical billing" is a highly specialized task requiring explicit knowledge of healthcare coding systems, insurance procedures, and regulations. It maps directly to concrete professional competencies (e.g., [Medical Billing]).
FINAL DECISION: RELEVANT

Text: "Pricing analysis"
Reasoning: "Pricing analysis" is a specific, concrete hard skill involving the evaluation of market trends, cost structures, and financial strategies. It is explicitly extractable.
FINAL DECISION: RELEVANT

Text: "Prior LIHTC recertification experience is necessary"
Reasoning: "LIHTC recertification" refers to a highly specific, technical domain process (Low-Income Housing Tax Credit). This requires specialized, mappable domain expertise and regulatory knowledge, making it a concrete skill.
FINAL DECISION: RELEVANT
"""


def build_gate_audit_messages(sentence: str) -> list[dict]:
    return [
        {"role": "system", "content": GATE_AUDIT_SYSTEM_PROMPT},
        {"role": "user", "content": f'Text: "{sentence}"\n\nReasoning and Decision:'},
    ]


def parse_gate_decision(text: str) -> int:
    """Parse the auditor's chain-of-thought reply into a 0/1 decision.

    Falls back to 1 (keep) if the model doesn't emit the expected marker --
    a conservative fallback: an unparseable audit should never silently drop
    a sentence the gate would have kept.
    """
    up = (text or "").upper()
    if "FINAL DECISION: NON-RELEVANT" in up:
        return 0
    if "FINAL DECISION: RELEVANT" in up:
        return 1
    return 1
