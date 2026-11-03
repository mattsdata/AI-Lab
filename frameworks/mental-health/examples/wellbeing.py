# ==================================================================================================
# 📜 ailab-mental-health :: examples :: wellbeing
# --------------------------------------------------------------------------------------------------
#  File         : frameworks/mental-health/examples/wellbeing.py
#  Author       : Mathew Varghese <removed@example.invalid>
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    A private command-line wellbeing journal: create a key, check in daily, read a gentle
#    summary, or score a PHQ-9 / GAD-7 questionnaire, all on your own computer.
#
#  🔧 WHAT IT DOES
#    key      -> generate an encryption key (keep it in a password manager)
#    checkin  -> add today's mood, sleep and an optional note to the encrypted journal
#    summary  -> last week vs the four weeks before, with support suggestions
#    phq9 / gad7 -> score answers from the official forms (0-3 each)
#
#  🧠 DESIGN NOTES
#    - The key is read from the AILAB_JOURNAL_KEY environment variable, never from a file in
#      the repository and never printed except when first generated.
#    - Safety alerts print crisis contacts before anything else.
#
#  USAGE
#    python frameworks/mental-health/examples/wellbeing.py key
#    export AILAB_JOURNAL_KEY="<the key>"
#    python frameworks/mental-health/examples/wellbeing.py checkin --mood 6 --note "ok day"
#    python frameworks/mental-health/examples/wellbeing.py summary
#    python frameworks/mental-health/examples/wellbeing.py phq9 1 0 1 2 0 1 0 0 0
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial demo: key, checkin, summary, phq9, gad7 commands.
# ==================================================================================================
"""Private wellbeing journal and screening scores (not a diagnosis)."""

# ========================================
# 📂 Imports
# ========================================
from __future__ import annotations

import argparse
import os
from datetime import date
from pathlib import Path

from ailab_mental_health import (
    CheckIn,
    FernetCipher,
    JournalStore,
    safety_message,
    score_gad7,
    score_phq9,
    wellbeing_summary,
)

DEFAULT_JOURNAL = Path.home() / ".ailab" / "journal.bin"


# ========================================
# 📂 Commands
# ========================================
def store(path: Path) -> JournalStore:
    key = os.environ.get("AILAB_JOURNAL_KEY")
    if not key:
        raise SystemExit("Set AILAB_JOURNAL_KEY first (run the 'key' command once to create one).")
    return JournalStore(path, FernetCipher(key.encode()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", type=Path, default=DEFAULT_JOURNAL)
    parser.add_argument("--region", default="UK", help="crisis contacts region: UK, IN, US")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("key")
    checkin = commands.add_parser("checkin")
    checkin.add_argument("--mood", type=int, required=True, help="1 (very low) to 10 (very good)")
    checkin.add_argument("--sleep", type=float)
    checkin.add_argument("--note", default="")
    commands.add_parser("summary")
    for name, items in (("phq9", 9), ("gad7", 7)):
        sub = commands.add_parser(name)
        sub.add_argument("answers", type=int, nargs=items)
    args = parser.parse_args()

    if args.command == "key":
        print(FernetCipher.generate_key().decode())
        print("Store this key safely (e.g. a password manager). Without it the journal is lost.")
    elif args.command == "checkin":
        store(args.journal).add(CheckIn(date.today(), args.mood, args.sleep, args.note))
        print(f"Check-in saved (encrypted) to {args.journal}")
    elif args.command == "summary":
        print(
            wellbeing_summary(
                store(args.journal).load(), as_of=date.today(), region=args.region
            ).text
        )
    else:
        result = score_phq9(args.answers) if args.command == "phq9" else score_gad7(args.answers)
        if result.safety_alert:
            print(safety_message(args.region), end="\n\n")
        print(f"{result.instrument}: {result.total}/{result.maximum} ({result.band})")
        print(result.note)


if __name__ == "__main__":
    main()
