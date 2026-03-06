"""
src/triage_engine.py
=====================
Phase 5 — Symptom Triage Engine (DEMO ONLY)

⚠️  SAFETY NOTICE  ⚠️
This module is a RESEARCH DEMONSTRATION and is NOT a medical device.
It does NOT diagnose, treat, or prescribe. All output is advisory only.
See docs/SAFETY.md for full disclaimer and escalation guidelines.

Architecture
------------
1. Rule-based emergency detector  — zero latency, always runs first.
   Flags life-threatening keywords → immediate "call emergency services" response.
   No LLM is consulted for flagged emergencies.

2. LLM wrapper (optional)         — Anthropic Claude via API.
   Constrained to three safe advice levels only. Structured JSON output.
   Falls back gracefully if no API key is set or API is unreachable.

3. Local fallback                 — pure Python, no API required.
   Keyword → advice mapping used when LLM unavailable.

4. Schema validation              — pydantic v2.
   Every response, LLM or fallback, passes through TriageResponse.
   Invalid LLM output is caught and replaced with a safe default.

Output contract (TriageResponse)
---------------------------------
{
  "advice_level":   "emergency" | "see_doctor" | "monitor",
  "advice_text":    str,          # human-readable guidance
  "escalate":       bool,         # True → show emergency numbers
  "disclaimer":     str,          # always present
  "source":         "rule" | "llm" | "fallback",
  "emergency_numbers": dict       # Moroccan emergency contacts
}
"""

from __future__ import annotations

import json
import logging
import os
import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# ── Safety constants ──────────────────────────────────────────────────────────

DISCLAIMER = (
    "⚠️ DEMO ONLY — This is not a medical device and does not provide "
    "medical diagnoses. Always consult a qualified healthcare professional. "
    "In an emergency, call the appropriate emergency services immediately."
)

EMERGENCY_NUMBERS = {
    "SAMU (medical emergency)": "15",
    "Police":                   "19",
    "Fire / Civil Protection":  "15",
    "Gendarmerie":              "177",
    "General Emergency":        "112",
}

EMERGENCY_ADVICE = (
    "🚨 EMERGENCY DETECTED — Call emergency services immediately (SAMU: 15 / 112). "
    "Do not wait. If possible, have someone stay with the patient until help arrives."
)

SEE_DOCTOR_ADVICE = (
    "Please consult a doctor or visit a clinic as soon as possible. "
    "Your symptoms warrant professional evaluation. "
    "Use the facility map to find the nearest healthcare centre."
)

MONITOR_ADVICE = (
    "Monitor your symptoms and rest. If symptoms worsen or new symptoms appear, "
    "consult a doctor. Stay hydrated and follow general health guidelines."
)


# ── Emergency keyword rules ───────────────────────────────────────────────────

# Patterns are matched case-insensitively against the full symptom text.
# Each tuple: (regex_pattern, human_readable_label)
# These are intentionally conservative — false positives here mean
# "unnecessarily escalated" which is always the safer error.

EMERGENCY_PATTERNS: list[tuple[str, str]] = [
    # Cardiac
    (r"\bchest\s+pain\b",                    "chest pain"),
    (r"\bchest\s+tight(ness|ening)?\b",      "chest tightness"),
    (r"\bpressure\s+(in|on)\s+(my\s+)?chest\b", "chest pressure"),
    (r"\bheart\s+attack\b",                  "heart attack"),
    (r"\bcardiac\b",                         "cardiac event"),
    # Respiratory
    (r"\bdifficulty\s+breath(ing)?\b",       "difficulty breathing"),
    (r"\bcan'?t\s+breath(e)?\b",             "unable to breathe"),
    (r"\bshortness\s+of\s+breath\b",         "shortness of breath"),
    (r"\bsob\b",                             "shortness of breath (SOB)"),
    (r"\brespiratory\s+(distress|failure)\b","respiratory distress"),
    # Stroke
    (r"\bstroke\b",                          "stroke"),
    (r"\bface\s+(drooping|droop)\b",         "facial drooping"),
    (r"\barm\s+weakness\b",                  "arm weakness"),
    (r"\bsudden\s+(numbness|weakness)\b",    "sudden numbness/weakness"),
    (r"\bslurred?\s+speech\b",               "slurred speech"),
    (r"\bsudden\s+(severe\s+)?headache\b",   "sudden severe headache"),
    (r"\bvision\s+(loss|changes?|problems?)\b","vision changes"),
    # Bleeding
    (r"\bsevere\s+bleeding\b",               "severe bleeding"),
    (r"\bheavy\s+bleeding\b",               "heavy bleeding"),
    (r"\bbleeding\s+(won'?t|doesn'?t)\s+stop\b", "uncontrolled bleeding"),
    (r"\bcoughing\s+blood\b",               "coughing blood"),
    (r"\bvomiting\s+blood\b",               "vomiting blood"),
    # Loss of consciousness
    (r"\bunconscious(ness)?\b",             "unconsciousness"),
    (r"\bpassed?\s+out\b",                  "loss of consciousness"),
    (r"\bfaint(ing|ed)?\b",                 "fainting"),
    (r"\bnot\s+respond(ing)?\b",            "not responding"),
    # Severe allergic
    (r"\banaphylaxis\b",                    "anaphylaxis"),
    (r"\bsevere\s+allergic\s+reaction\b",   "severe allergic reaction"),
    (r"\bthroat\s+(closing|swelling)\b",    "throat swelling"),
    # Trauma
    (r"\bsevere\s+(injury|injuries)\b",     "severe injury"),
    (r"\bhead\s+injury\b",                  "head injury"),
    (r"\bseizure\b",                        "seizure"),
    (r"\bconvulsion\b",                     "convulsion"),
]

_COMPILED_PATTERNS = [
    (re.compile(pat, re.IGNORECASE), label)
    for pat, label in EMERGENCY_PATTERNS
]


# ── Schema ────────────────────────────────────────────────────────────────────

class AdviceLevel(str, Enum):
    EMERGENCY  = "emergency"
    SEE_DOCTOR = "see_doctor"
    MONITOR    = "monitor"


class TriageResponse(BaseModel):
    advice_level:      AdviceLevel
    advice_text:       str
    escalate:          bool
    disclaimer:        str = DISCLAIMER
    source:            str = Field(default="fallback", pattern=r"^(rule|llm|fallback)$")
    triggered_rules:   list[str] = Field(default_factory=list)
    emergency_numbers: dict[str, str] = Field(default_factory=dict)

    @field_validator("advice_text")
    @classmethod
    def no_diagnosis_language(cls, v: str) -> str:
        """
        Reject any advice text that contains diagnosis-like language.
        This is a last-resort safety net — the LLM prompt already prohibits
        this, but we validate again at the schema level.
        """
        forbidden = [
            r"\byou have\b", r"\byou are diagnosed\b", r"\byour diagnosis\b",
            r"\bdiagnosis is\b", r"\btest (shows|indicates|confirms)\b",
        ]
        for pattern in forbidden:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError(
                    f"Advice text contains prohibited diagnostic language: '{pattern}'"
                )
        return v

    @field_validator("escalate")
    @classmethod
    def emergency_must_escalate(cls, v: bool, info: Any) -> bool:
        """Emergency advice level must always set escalate=True."""
        data = info.data
        if data.get("advice_level") == AdviceLevel.EMERGENCY and not v:
            return True  # force escalate rather than reject
        return v

    model_config = {"use_enum_values": True}


# ── Rule-based detector ───────────────────────────────────────────────────────

def detect_emergency(text: str) -> tuple[bool, list[str]]:
    """
    Run all emergency patterns against the symptom text.

    Returns
    -------
    (is_emergency, triggered_labels)
    """
    triggered = []
    for compiled, label in _COMPILED_PATTERNS:
        if compiled.search(text):
            triggered.append(label)
    return bool(triggered), triggered


def rule_based_triage(symptom_text: str) -> TriageResponse:
    """
    Emergency-only rule path. Called when detect_emergency() fires.
    Never consults the LLM — latency and availability don't matter here.
    """
    _, triggered = detect_emergency(symptom_text)
    return TriageResponse(
        advice_level=AdviceLevel.EMERGENCY,
        advice_text=EMERGENCY_ADVICE,
        escalate=True,
        source="rule",
        triggered_rules=triggered,
        emergency_numbers=EMERGENCY_NUMBERS,
    )


# ── LLM wrapper ───────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a safe, conservative medical triage assistant for a healthcare
access research project in Morocco. You are NOT a doctor. You do NOT diagnose.

STRICT RULES:
1. Never diagnose any condition.
2. Never prescribe medication or dosage.
3. Never claim certainty about what is wrong.
4. Only use one of three advice levels: "emergency", "see_doctor", "monitor".
5. Keep advice_text under 80 words.
6. Do not repeat symptoms back as if confirming them.

ADVICE LEVEL GUIDE:
- "emergency"  → Symptoms could be life-threatening or time-critical.
- "see_doctor" → Symptoms warrant professional evaluation within 24–48 hours.
- "monitor"    → Symptoms are mild; rest, hydrate, watch for changes.

RESPONSE FORMAT — return ONLY valid JSON, no markdown, no preamble:
{
  "advice_level": "emergency" | "see_doctor" | "monitor",
  "advice_text": "...",
  "escalate": true | false
}"""


def _call_llm(symptom_text: str, api_key: str) -> dict:
    """
    Call Anthropic Claude and return the parsed JSON dict.
    Raises on API error or invalid JSON — caller handles fallback.
    """
    import anthropic  # lazy import — optional dependency

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",   # fast, cheap, sufficient for triage
        max_tokens=256,
        system=_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": (
                f"Patient symptoms (Morocco, research demo):\n{symptom_text}\n\n"
                "Respond with JSON only."
            ),
        }],
    )
    raw = message.content[0].text.strip()

    # Strip accidental markdown fences
    raw = re.sub(r"^```json\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"```$", "", raw)

    return json.loads(raw)


def llm_triage(symptom_text: str, api_key: str) -> TriageResponse:
    """
    LLM path — called only when no emergency rule fires AND an API key exists.
    Falls back to local triage on any failure.
    """
    try:
        raw = _call_llm(symptom_text, api_key)

        # Build and validate via pydantic
        response = TriageResponse(
            advice_level=raw["advice_level"],
            advice_text=raw["advice_text"],
            escalate=raw.get("escalate", raw["advice_level"] == "emergency"),
            source="llm",
            emergency_numbers=EMERGENCY_NUMBERS if raw.get("escalate") else {},
        )
        logger.info("LLM triage succeeded: level=%s", response.advice_level)
        return response

    except Exception as exc:
        logger.warning("LLM triage failed (%s) — using local fallback.", exc)
        return local_fallback_triage(symptom_text)


# ── Local fallback ────────────────────────────────────────────────────────────

# Keyword → (AdviceLevel, brief_reason) mapping.
# Conservative: when in doubt, escalate to see_doctor.
_FALLBACK_RULES: list[tuple[str, AdviceLevel, str]] = [
    # see_doctor triggers
    ("fever",         AdviceLevel.SEE_DOCTOR, "persistent fever warrants evaluation"),
    ("high temperature", AdviceLevel.SEE_DOCTOR, "high temperature warrants evaluation"),
    ("vomiting",      AdviceLevel.SEE_DOCTOR, "vomiting may need medical attention"),
    ("diarrhea",      AdviceLevel.SEE_DOCTOR, "diarrhea may need medical attention"),
    ("diarrhoea",     AdviceLevel.SEE_DOCTOR, "diarrhoea may need medical attention"),
    ("rash",          AdviceLevel.SEE_DOCTOR, "skin rash should be assessed"),
    ("infection",     AdviceLevel.SEE_DOCTOR, "possible infection needs assessment"),
    ("pain",          AdviceLevel.SEE_DOCTOR, "pain should be evaluated by a doctor"),
    ("bleeding",      AdviceLevel.SEE_DOCTOR, "bleeding requires professional assessment"),
    ("swelling",      AdviceLevel.SEE_DOCTOR, "swelling should be assessed"),
    ("injury",        AdviceLevel.SEE_DOCTOR, "injury needs professional evaluation"),
    ("wound",         AdviceLevel.SEE_DOCTOR, "wound needs professional care"),
    ("pregnant",      AdviceLevel.SEE_DOCTOR, "pregnancy symptoms need medical supervision"),
    ("child",         AdviceLevel.SEE_DOCTOR, "symptoms in children need prompt evaluation"),
    ("baby",          AdviceLevel.SEE_DOCTOR, "infant symptoms need prompt evaluation"),
    ("elderly",       AdviceLevel.SEE_DOCTOR, "symptoms in elderly patients need evaluation"),
    # monitor triggers (mild, self-limiting)
    ("cold",          AdviceLevel.MONITOR, "common cold — rest and monitor"),
    ("cough",         AdviceLevel.MONITOR, "mild cough — rest and monitor"),
    ("runny nose",    AdviceLevel.MONITOR, "runny nose — rest and monitor"),
    ("sneeze",        AdviceLevel.MONITOR, "sneezing — rest and monitor"),
    ("tired",         AdviceLevel.MONITOR, "fatigue — rest and monitor"),
    ("fatigue",       AdviceLevel.MONITOR, "fatigue — rest and monitor"),
    ("headache",      AdviceLevel.MONITOR, "mild headache — rest, hydrate, monitor"),
    ("sore throat",   AdviceLevel.MONITOR, "mild sore throat — rest and monitor"),
    ("nausea",        AdviceLevel.MONITOR, "nausea — rest, hydrate, monitor"),
]

_ADVICE_TEXTS = {
    AdviceLevel.EMERGENCY:  EMERGENCY_ADVICE,
    AdviceLevel.SEE_DOCTOR: SEE_DOCTOR_ADVICE,
    AdviceLevel.MONITOR:    MONITOR_ADVICE,
}


def local_fallback_triage(symptom_text: str) -> TriageResponse:
    """
    Pure-Python keyword triage. No API required.
    Used when no API key is set or when the LLM call fails.

    Priority: see_doctor > monitor > (no match → see_doctor as safe default).
    First matching keyword wins within its tier; see_doctor always beats monitor.
    """
    text_lower = symptom_text.lower()
    matched_level = None  # no match yet

    for keyword, level, _ in _FALLBACK_RULES:
        if keyword in text_lower:
            if matched_level is None:
                matched_level = level
            elif level == AdviceLevel.SEE_DOCTOR and matched_level == AdviceLevel.MONITOR:
                # escalate to see_doctor — never de-escalate
                matched_level = level

    if matched_level is None:
        matched_level = AdviceLevel.SEE_DOCTOR  # conservative default when nothing matches

    escalate = matched_level == AdviceLevel.EMERGENCY
    return TriageResponse(
        advice_level=matched_level,
        advice_text=_ADVICE_TEXTS[matched_level],
        escalate=escalate,
        source="fallback",
        emergency_numbers=EMERGENCY_NUMBERS if escalate else {},
    )


# ── Public API ────────────────────────────────────────────────────────────────

def triage(symptom_text: str) -> TriageResponse:
    """
    Main entry point. Always safe to call — never raises.

    Decision tree
    -------------
    1. Emergency rules → immediate EMERGENCY response (no LLM).
    2. ANTHROPIC_API_KEY set → LLM triage (with fallback on failure).
    3. No API key → local fallback triage.

    Parameters
    ----------
    symptom_text : str
        Free-text symptom description from the user.

    Returns
    -------
    TriageResponse (always valid, never raises)
    """
    if not symptom_text or not symptom_text.strip():
        return TriageResponse(
            advice_level=AdviceLevel.MONITOR,
            advice_text="Please describe your symptoms so we can provide guidance.",
            escalate=False,
            source="fallback",
        )

    # Step 1 — emergency rules (always first, always fast)
    is_emergency, _ = detect_emergency(symptom_text)
    if is_emergency:
        return rule_based_triage(symptom_text)

    # Step 2 — LLM if API key available
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if api_key and api_key != "your_anthropic_api_key_here":
        return llm_triage(symptom_text, api_key)

    # Step 3 — local fallback
    return local_fallback_triage(symptom_text)
