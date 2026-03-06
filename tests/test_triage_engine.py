"""
tests/test_triage_engine.py
============================
Unit tests for Phase 5 triage engine.

Tests cover:
- Rule-based emergency detection (all symptom categories)
- Local fallback triage (keyword mapping)
- Schema validation (TriageResponse pydantic model)
- Safety invariants (emergency always escalates, no diagnosis language)
- LLM path (mocked — no real API calls in CI)
- Edge cases (empty input, unknown symptoms)
"""

import os
import unittest
from unittest.mock import MagicMock, patch

import pytest

from src.triage_engine import (
    AdviceLevel,
    TriageResponse,
    detect_emergency,
    local_fallback_triage,
    rule_based_triage,
    triage,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Emergency detection — rule layer
# ═══════════════════════════════════════════════════════════════════════════════

class TestEmergencyDetection:

    # ── Cardiac ───────────────────────────────────────────────────────────────
    @pytest.mark.parametrize("text", [
        "I have chest pain",
        "There is pressure on my chest",
        "I feel chest tightness",
        "I think I'm having a heart attack",
    ])
    def test_cardiac_detected(self, text):
        is_emerg, triggered = detect_emergency(text)
        assert is_emerg, f"Should detect emergency in: {text!r}"
        assert len(triggered) > 0

    # ── Respiratory ───────────────────────────────────────────────────────────
    @pytest.mark.parametrize("text", [
        "I have difficulty breathing",
        "I can't breathe properly",
        "Shortness of breath",
        "I have severe SOB",
        "respiratory distress",
    ])
    def test_respiratory_detected(self, text):
        is_emerg, _ = detect_emergency(text)
        assert is_emerg, f"Should detect emergency in: {text!r}"

    # ── Stroke ────────────────────────────────────────────────────────────────
    @pytest.mark.parametrize("text", [
        "I think I'm having a stroke",
        "Face drooping on one side",
        "sudden arm weakness",
        "slurred speech",
        "sudden severe headache",
        "sudden vision loss",
    ])
    def test_stroke_detected(self, text):
        is_emerg, _ = detect_emergency(text)
        assert is_emerg, f"Should detect emergency in: {text!r}"

    # ── Bleeding ──────────────────────────────────────────────────────────────
    @pytest.mark.parametrize("text", [
        "severe bleeding from a wound",
        "heavy bleeding won't stop",
        "I'm coughing blood",
        "vomiting blood",
    ])
    def test_bleeding_detected(self, text):
        is_emerg, _ = detect_emergency(text)
        assert is_emerg, f"Should detect emergency in: {text!r}"

    # ── Loss of consciousness ─────────────────────────────────────────────────
    @pytest.mark.parametrize("text", [
        "patient is unconscious",
        "he passed out",
        "not responding to anything",
        "having a seizure",
        "convulsions",
    ])
    def test_unconsciousness_detected(self, text):
        is_emerg, _ = detect_emergency(text)
        assert is_emerg, f"Should detect emergency in: {text!r}"

    # ── Non-emergency — must NOT fire ─────────────────────────────────────────
    @pytest.mark.parametrize("text", [
        "I have a mild headache",
        "runny nose and slight fever",
        "feeling tired and have a cough",
        "sore throat for two days",
        "mild stomach ache",
        "",
    ])
    def test_non_emergency_not_detected(self, text):
        is_emerg, triggered = detect_emergency(text)
        assert not is_emerg, f"Should NOT detect emergency in: {text!r} (got {triggered})"

    def test_case_insensitive(self):
        is_emerg, _ = detect_emergency("CHEST PAIN AND DIFFICULTY BREATHING")
        assert is_emerg

    def test_returns_trigger_labels(self):
        _, triggered = detect_emergency("chest pain and difficulty breathing")
        assert isinstance(triggered, list)
        assert len(triggered) >= 2


# ═══════════════════════════════════════════════════════════════════════════════
# Rule-based triage response
# ═══════════════════════════════════════════════════════════════════════════════

class TestRuleBasedTriage:

    def test_returns_triage_response(self):
        r = rule_based_triage("chest pain")
        assert isinstance(r, TriageResponse)

    def test_level_is_emergency(self):
        r = rule_based_triage("I can't breathe")
        assert r.advice_level == AdviceLevel.EMERGENCY

    def test_escalate_is_true(self):
        r = rule_based_triage("severe bleeding")
        assert r.escalate is True

    def test_source_is_rule(self):
        r = rule_based_triage("stroke symptoms")
        assert r.source == "rule"

    def test_emergency_numbers_present(self):
        r = rule_based_triage("chest pain")
        assert "SAMU (medical emergency)" in r.emergency_numbers
        assert r.emergency_numbers["SAMU (medical emergency)"] == "15"

    def test_triggered_rules_populated(self):
        r = rule_based_triage("chest pain and difficulty breathing")
        assert len(r.triggered_rules) >= 2

    def test_disclaimer_always_present(self):
        r = rule_based_triage("chest pain")
        assert len(r.disclaimer) > 20


# ═══════════════════════════════════════════════════════════════════════════════
# Local fallback triage
# ═══════════════════════════════════════════════════════════════════════════════

class TestLocalFallbackTriage:

    @pytest.mark.parametrize("text,expected_level", [
        ("I have a high fever",         AdviceLevel.SEE_DOCTOR),
        ("persistent vomiting",         AdviceLevel.SEE_DOCTOR),
        ("skin rash on my arm",         AdviceLevel.SEE_DOCTOR),
        ("possible infection",          AdviceLevel.SEE_DOCTOR),
        ("I have pain in my back",      AdviceLevel.SEE_DOCTOR),
        ("my child is sick",            AdviceLevel.SEE_DOCTOR),
        ("runny nose and cough",        AdviceLevel.MONITOR),
        ("feeling tired today",         AdviceLevel.MONITOR),
        ("mild headache this morning",  AdviceLevel.MONITOR),
        ("slight sore throat",          AdviceLevel.MONITOR),
        ("nausea after eating",         AdviceLevel.MONITOR),
        ("I feel fatigued",             AdviceLevel.MONITOR),
    ])
    def test_keyword_routing(self, text, expected_level):
        r = local_fallback_triage(text)
        assert r.advice_level == expected_level, (
            f"'{text}' → expected {expected_level}, got {r.advice_level}"
        )

    def test_unknown_symptom_defaults_see_doctor(self):
        r = local_fallback_triage("xyzzy unknown symptom 12345")
        assert r.advice_level == AdviceLevel.SEE_DOCTOR

    def test_source_is_fallback(self):
        r = local_fallback_triage("cough")
        assert r.source == "fallback"

    def test_no_escalate_on_monitor(self):
        r = local_fallback_triage("mild headache")
        assert r.escalate is False

    def test_no_emergency_numbers_on_monitor(self):
        r = local_fallback_triage("tired")
        assert r.emergency_numbers == {}


# ═══════════════════════════════════════════════════════════════════════════════
# Schema validation — TriageResponse
# ═══════════════════════════════════════════════════════════════════════════════

class TestTriageResponseSchema:

    def test_valid_construction(self):
        r = TriageResponse(
            advice_level=AdviceLevel.SEE_DOCTOR,
            advice_text="Please consult a doctor.",
            escalate=False,
            source="fallback",
        )
        assert r.advice_level == AdviceLevel.SEE_DOCTOR

    def test_invalid_advice_level_raises(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            TriageResponse(
                advice_level="diagnose_condition",  # not a valid level
                advice_text="You have flu.",
                escalate=False,
                source="fallback",
            )

    def test_invalid_source_raises(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            TriageResponse(
                advice_level=AdviceLevel.MONITOR,
                advice_text="Monitor symptoms.",
                escalate=False,
                source="gpt4",  # not allowed
            )

    def test_diagnosis_language_rejected(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            TriageResponse(
                advice_level=AdviceLevel.SEE_DOCTOR,
                advice_text="You have influenza. You are diagnosed with flu.",
                escalate=False,
                source="llm",
            )

    def test_emergency_forces_escalate(self):
        """Emergency level with escalate=False should be auto-corrected to True."""
        r = TriageResponse(
            advice_level=AdviceLevel.EMERGENCY,
            advice_text="Call emergency services.",
            escalate=False,  # should be forced to True
            source="rule",
        )
        assert r.escalate is True

    def test_disclaimer_defaults_correctly(self):
        r = TriageResponse(
            advice_level=AdviceLevel.MONITOR,
            advice_text="Rest and monitor.",
            escalate=False,
            source="fallback",
        )
        assert "not a medical device" in r.disclaimer.lower() or "demo" in r.disclaimer.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# Main triage() entry point
# ═══════════════════════════════════════════════════════════════════════════════

class TestTriageEntryPoint:

    def test_empty_input_returns_monitor(self):
        r = triage("")
        assert r.advice_level == AdviceLevel.MONITOR

    def test_whitespace_only_returns_monitor(self):
        r = triage("   \n\t  ")
        assert r.advice_level == AdviceLevel.MONITOR

    def test_emergency_bypasses_llm(self):
        """Emergency rules must fire before any LLM call."""
        with patch("src.triage_engine._call_llm") as mock_llm:
            r = triage("severe chest pain and can't breathe")
        mock_llm.assert_not_called()
        assert r.advice_level == AdviceLevel.EMERGENCY

    def test_no_api_key_uses_fallback(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}, clear=False):
            r = triage("I have a fever and cough")
        assert r.source in ("fallback", "rule")

    def test_llm_called_when_api_key_set(self):
        """With a valid-looking key and no emergency, LLM path is taken."""
        mock_response = {
            "advice_level": "see_doctor",
            "advice_text": "Please consult a healthcare professional.",
            "escalate": False,
        }
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-fake-key"}, clear=False):
            with patch("src.triage_engine._call_llm", return_value=mock_response):
                r = triage("I have a moderate fever for three days")
        assert r.source == "llm"
        assert r.advice_level == AdviceLevel.SEE_DOCTOR

    def test_llm_failure_falls_back(self):
        """If LLM call raises, fallback is used transparently."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-fake-key"}, clear=False):
            with patch("src.triage_engine._call_llm", side_effect=ConnectionError("API down")):
                r = triage("I have a rash on my arm")
        assert r.source == "fallback"

    def test_llm_invalid_json_falls_back(self):
        """Malformed LLM JSON triggers fallback, not a crash."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-fake-key"}, clear=False):
            with patch("src.triage_engine._call_llm", side_effect=ValueError("bad json")):
                r = triage("vomiting since yesterday")
        assert r.source == "fallback"
        assert isinstance(r, TriageResponse)

    def test_always_returns_triage_response(self):
        """triage() must never raise — always return a valid TriageResponse."""
        test_inputs = [
            "chest pain",
            "mild cold",
            "xyzzy unknown",
            "",
            "a" * 800,
        ]
        for text in test_inputs:
            r = triage(text)
            assert isinstance(r, TriageResponse), f"Failed for input: {text[:40]!r}"


# ═══════════════════════════════════════════════════════════════════════════════
# Safety invariants
# ═══════════════════════════════════════════════════════════════════════════════

class TestSafetyInvariants:

    def test_emergency_always_has_disclaimer(self):
        r = triage("chest pain and difficulty breathing")
        assert len(r.disclaimer) > 0

    def test_every_response_has_disclaimer(self):
        for text in ["mild cough", "fever", "chest pain", ""]:
            r = triage(text)
            assert r.disclaimer, f"Missing disclaimer for input: {text!r}"

    def test_emergency_numbers_in_emergency_response(self):
        r = triage("I can't breathe and have chest pain")
        assert r.emergency_numbers
        assert "15" in r.emergency_numbers.values() or "112" in r.emergency_numbers.values()

    def test_no_diagnosis_in_fallback_advice(self):
        """Fallback advice texts must not contain diagnostic language."""
        for text in ["fever", "pain", "cough", "rash", "infection"]:
            r = local_fallback_triage(text)
            forbidden_phrases = ["you have", "you are diagnosed", "diagnosis is"]
            for phrase in forbidden_phrases:
                assert phrase not in r.advice_text.lower(), (
                    f"Diagnostic language '{phrase}' found in fallback advice for '{text}'"
                )
