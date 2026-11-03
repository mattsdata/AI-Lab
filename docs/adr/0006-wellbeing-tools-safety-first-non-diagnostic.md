# ADR-0006: Wellbeing tools are safety-first, private and non-diagnostic

- **Status:** Accepted
- **Date:** 2026-11-03
- **Author:** Mathew Varghese

## Context

Software that touches mental health can help people notice patterns and seek support early.
It can also cause harm: a wrong label, a missed sign of risk, or a leak of the most sensitive
personal data a person has. Engineering choices here are ethical choices.

## Decision

1. **Non-diagnostic.** The framework scores validated screening questionnaires with their
   published cut-offs and describes a person's own entries. It never names a condition. Every
   result says it is not a diagnosis and suggests a GP or clinician. A test scans summaries for
   diagnostic terms.
2. **Safety overrides everything.** Any positive answer to PHQ-9 item 9, or crisis language in
   a recent note, raises a safety alert and puts regional crisis contacts first in the output.
   The check errs towards showing help. Contacts are data, re-verified before each release.
3. **Private by design.** Journals live on the person's device, encrypted at rest with
   authenticated encryption (Fernet). There is no plain-text mode. Note text is excluded from
   reprs and never logged. Nothing is sent over the network.
4. **No invented instruments.** Only validated instruments with published scoring are used;
   question wording is not reproduced, and licensing is checked before distribution.
5. **Clinical review before real use.** Thresholds (for example 14 days of low mood) are
   parameters so that a clinician can set them; the defaults are conservative placeholders.

## Consequences

**Positive**
- Users get pattern awareness and a route to help without being labelled.
- The safety behaviour is in one module and covered by tests, so it cannot drift per feature.
- A lost device does not expose the journal; a wrong key fails loudly.

**Trade-offs**
- Losing the encryption key means losing the journal; that is the price of real privacy.
- Phrase-based crisis detection will produce false alarms; that direction of error is chosen.
- This is a portfolio and research framework, not a medical device; deploying it to patients
  would require clinical governance and regulatory review (e.g. MHRA in the UK).
