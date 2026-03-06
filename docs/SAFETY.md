# Safety & Disclaimer — Morocco Healthcare Triage Bot

> **Read this document before using, sharing, or deploying the triage feature.**

---

## 1. This Is Not a Medical Device

The symptom triage feature in this project is a **research and portfolio demonstration**.
It has **not** been evaluated, approved, or certified by any medical regulatory body,
including but not limited to:

- Morocco's Ministry of Health (Ministère de la Santé)
- The European Medicines Agency (EMA)
- The U.S. Food and Drug Administration (FDA)
- The World Health Organization (WHO)

**It must not be used to make real medical decisions.**

---

## 2. This Is Not Diagnostic

The triage engine does **not**:

- Diagnose any medical condition
- Identify the cause of any symptom
- Confirm or rule out any disease, illness, or injury
- Replace clinical examination, laboratory tests, or imaging
- Substitute for a qualified healthcare professional's judgment

All output is limited to one of three advisory levels:

| Level | Meaning |
|-------|---------|
| 🚨 **Emergency** | Seek emergency care immediately — call 15 or 112 |
| 🩺 **See a doctor** | Consult a healthcare professional within 24–48 hours |
| 👁 **Monitor** | Rest, hydrate, and watch for changes |

These levels are **not diagnoses**. They are conservative suggestions based on
keyword pattern matching and/or a language model prompt with strict safety constraints.

---

## 3. Research Demo — Intended Use

This feature was built to demonstrate:

- Safe LLM integration patterns in healthcare-adjacent applications
- Rule-based emergency escalation as a first-pass safety layer
- Schema-validated structured output from language models
- Graceful local fallback when no API is available

**Intended audience:** developers, researchers, portfolio reviewers.  
**Not intended for:** patients, caregivers, clinicians, or any real-world triage use.

---

## 4. Emergency Escalation Guidelines

The rule-based layer automatically escalates the following symptom categories
to **EMERGENCY** level. This path bypasses the LLM entirely for speed and reliability.

### 4.1 Cardiac Symptoms
- Chest pain, pressure, or tightness
- Suspected heart attack

### 4.2 Respiratory Symptoms
- Difficulty breathing or shortness of breath
- Inability to breathe
- Respiratory distress or failure

### 4.3 Stroke Symptoms (FAST criteria)
- **F**ace drooping
- **A**rm weakness or sudden numbness
- **S**peech slurred or sudden difficulty speaking
- **T**ime — sudden severe headache, vision loss, confusion

### 4.4 Severe Bleeding
- Heavy or uncontrolled bleeding
- Bleeding that will not stop
- Coughing or vomiting blood

### 4.5 Loss of Consciousness
- Unconsciousness, unresponsiveness
- Fainting or passing out
- Seizures or convulsions

### 4.6 Severe Allergic Reaction
- Anaphylaxis
- Throat closing or swelling

---

## 5. Moroccan Emergency Contacts

| Service | Number |
|---------|--------|
| SAMU (Medical Emergency) | **15** |
| Police Secours | **19** |
| Gendarmerie Royale | **177** |
| Protection Civile (Fire) | **15** |
| International Emergency | **112** |

---

## 6. Data & Privacy

- Symptom text entered into the triage widget is sent to the Anthropic API
  when an `ANTHROPIC_API_KEY` is configured.
- In local fallback mode (no API key), no data leaves the machine.
- No symptom data is stored, logged, or persisted by this application.
- Do not enter real patient data into this demo system.

---

## 7. Liability

The authors of this project accept **no liability** for any harm arising from the
use of this triage demonstration. By using this feature, you acknowledge that:

1. It is a demo and not a certified medical tool.
2. You will not use it to make real medical decisions.
3. In a real emergency you will call emergency services immediately.

---

## 8. Contributing Safety Improvements

If you identify a symptom pattern that should trigger emergency escalation
but does not, please open a GitHub issue with the label `safety`.

All safety-related issues are treated as highest priority.

---

*Last updated: Phase 5 implementation*  
*Maintainer: Morocco Healthcare Optimization Project*
