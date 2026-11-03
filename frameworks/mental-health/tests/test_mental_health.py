# ==================================================================================================
# 📜 ailab-mental-health :: tests
# --------------------------------------------------------------------------------------------------
#  File         : frameworks/mental-health/tests/test_mental_health.py
#  Author       : Mathew Varghese <removed@example.invalid>
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    Proves the safety rules hold: published cut-offs exactly, the self-harm item always raises
#    an alert, crisis language always surfaces help first, data is unreadable without the key,
#    note text never leaks into reprs, and summaries never use diagnostic labels.
#
#  🧠 TESTING APPROACH
#    Boundary values for every band; real Fernet encryption on disk via pytest's tmp_path;
#    synthetic check-in histories for weekly comparisons and sustained low mood.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial test suite for ailab-mental-health.
# ==================================================================================================

# ========================================
# 📂 Imports
# ========================================
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

from ailab_mental_health import (
    CRISIS_RESOURCES,
    NOT_A_DIAGNOSIS,
    CheckIn,
    FernetCipher,
    JournalStore,
    mentions_crisis,
    safety_message,
    score_gad7,
    score_phq9,
    wellbeing_summary,
)

TODAY = date(2026, 11, 3)
DIAGNOSTIC_TERMS = (
    "depression",
    "depressed",
    "disorder",
    "anxiety disorder",
    "diagnos",
    "you have",
)


def history(moods: list[int], *, notes: dict[int, str] | None = None) -> list[CheckIn]:
    """Check-ins ending today; moods[-1] is today."""
    notes = notes or {}
    start = TODAY - timedelta(days=len(moods) - 1)
    return [
        CheckIn(start + timedelta(days=i), m, 7.0, notes.get(i, "")) for i, m in enumerate(moods)
    ]


# ========================================
# 📂 Screening Tests
# ========================================
@pytest.mark.parametrize(
    ("total", "band"),
    [
        (0, "minimal"),
        (4, "minimal"),
        (5, "mild"),
        (9, "mild"),
        (10, "moderate"),
        (14, "moderate"),
        (15, "moderately severe"),
        (19, "moderately severe"),
        (20, "severe"),
        (24, "severe"),
    ],
)
def test_phq9_bands_match_published_cut_offs(total: int, band: str) -> None:
    answers = [min(3, max(0, total - 3 * i)) for i in range(8)] + [0]  # item 9 kept at zero
    result = score_phq9(answers)
    assert (result.total, result.band, result.safety_alert) == (total, band, False)
    assert result.note == NOT_A_DIAGNOSIS


def test_phq9_maximum_score_includes_a_safety_alert() -> None:
    result = score_phq9([3] * 9)  # 27 is only reachable with item 9 > 0
    assert (result.total, result.band, result.safety_alert) == (27, "severe", True)


def test_self_harm_item_always_raises_an_alert() -> None:
    result = score_phq9([0, 0, 0, 0, 0, 0, 0, 0, 1])
    assert result.band == "minimal" and result.safety_alert  # total 1, still an alert


@pytest.mark.parametrize(
    ("total", "band"),
    [(4, "minimal"), (5, "mild"), (10, "moderate"), (15, "severe"), (21, "severe")],
)
def test_gad7_bands(total: int, band: str) -> None:
    answers = [min(3, max(0, total - 3 * i)) for i in range(7)]
    result = score_gad7(answers)
    assert (result.total, result.band, result.safety_alert) == (total, band, False)


@pytest.mark.parametrize("bad", [[0] * 8, [0] * 8 + [4], [0] * 8 + [-1], [0] * 8 + [1.5]])
def test_screening_validation(bad: list) -> None:  # type: ignore[type-arg]
    with pytest.raises(ValueError):
        score_phq9(bad)
    with pytest.raises(ValueError):
        score_gad7([0] * 6)


# ========================================
# 📂 Safety Tests
# ========================================
@pytest.mark.parametrize(
    "text",
    [
        "I keep thinking about suicide",
        "I want to kill myself",
        "sometimes I self-harm",
        "I don't want to be alive",
        "everyone would be better off dead without me",
        "I want to end it all",
    ],
)
def test_crisis_language_is_detected(text: str) -> None:
    assert mentions_crisis(text)


@pytest.mark.parametrize(
    "text", ["Had a long walk, felt calm", "Killer workout today", "deadline stress"]
)
def test_ordinary_text_is_not_flagged(text: str) -> None:
    assert not mentions_crisis(text)


def test_safety_message_lists_regional_contacts() -> None:
    uk = safety_message("uk")
    assert "Samaritans: 116 123" in uk and "999" in uk
    assert "14416" in safety_message("IN") and "988" in safety_message("US")
    with pytest.raises(ValueError, match="add them before use"):
        safety_message("ZZ")
    assert all(r[0].name == "Emergency services" for r in CRISIS_RESOURCES.values())


# ========================================
# 📂 Journal Tests
# ========================================
def test_journal_is_encrypted_at_rest(tmp_path: Path) -> None:
    key = FernetCipher.generate_key()
    store = JournalStore(tmp_path / "journal.bin", FernetCipher(key))
    assert store.load() == []
    store.add(CheckIn(TODAY, 6, 7.5, "private note about work"))
    store.add(CheckIn(TODAY - timedelta(days=1), 4, 6.0))
    store.add(CheckIn(TODAY, 7, 8.0, "updated"))  # same day replaces
    raw = (tmp_path / "journal.bin").read_bytes()
    assert b"private" not in raw and b"mood" not in raw  # nothing readable on disk
    entries = store.load()
    assert [(e.day, e.mood) for e in entries] == [(TODAY - timedelta(days=1), 4), (TODAY, 7)]
    with pytest.raises(ValueError, match="wrong key"):
        JournalStore(tmp_path / "journal.bin", FernetCipher(FernetCipher.generate_key())).load()


def test_notes_never_appear_in_repr_and_entries_are_validated() -> None:
    entry = CheckIn(TODAY, 5, 7.0, "very private")
    assert "very private" not in repr(entry)
    with pytest.raises(ValueError):
        CheckIn(TODAY, 0)
    with pytest.raises(ValueError):
        CheckIn(TODAY, 5, 25.0)


def test_missing_cryptography_gives_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "cryptography.fernet", None)
    with pytest.raises(ImportError, match=r"ailab-mental-health\[crypto\]"):
        FernetCipher(b"x")


# ========================================
# 📂 Summary Tests
# ========================================
def test_summary_compares_weeks_without_labels() -> None:
    summary = wellbeing_summary(history([7] * 28 + [4] * 7), as_of=TODAY)
    assert summary.recent_mood == 4.0 and summary.earlier_mood == 7.0
    assert "lower than the 4 weeks before (7.0)" in summary.text
    assert "slept 7.0 hours" in summary.text and "not a medical assessment" in summary.text
    assert not any(term in summary.text.lower() for term in DIAGNOSTIC_TERMS)
    steady = wellbeing_summary(history([6] * 35), as_of=TODAY)
    assert "about the same" in steady.text
    better = wellbeing_summary(history([4] * 28 + [7] * 7), as_of=TODAY)
    assert "higher than" in better.text


def test_sustained_low_mood_suggests_support() -> None:
    summary = wellbeing_summary(history([6] * 10 + [2] * 14), as_of=TODAY)
    assert summary.low_streak_days == 14 and "talk to your GP" in summary.text
    broken = wellbeing_summary(history([2] * 13 + [6] + [2]), as_of=TODAY)
    assert broken.low_streak_days == 1 and "talk to your GP" not in broken.text


def test_crisis_note_puts_help_first() -> None:
    entries = history([5] * 7, notes={6: "I don't want to be alive anymore"})
    summary = wellbeing_summary(entries, as_of=TODAY, region="UK")
    assert summary.crisis_language
    assert summary.text.startswith("You don't have to go through this alone")
    assert "116 123" in summary.text


def test_no_recent_entries_and_future_entries_ignored() -> None:
    old = [CheckIn(TODAY - timedelta(days=60), 5)]
    future = [CheckIn(TODAY + timedelta(days=1), 1, note="suicide")]
    summary = wellbeing_summary(old + future, as_of=TODAY)
    assert summary.recent_mood is None and not summary.crisis_language
    assert "No check-ins in the last 7 days" in summary.text
