# ==================================================================================================
# 📜 ailab-mental-health :: journal
# --------------------------------------------------------------------------------------------------
#  File         : frameworks/mental-health/src/ailab_mental_health/journal.py
#  Author       : Mathew Varghese <removed@example.invalid>
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    A private daily check-in: mood, sleep and an optional note, stored only on the person's own
#    device and encrypted at rest. Wellbeing data is among the most sensitive personal data there
#    is; the storage design treats it that way by default.
#
#  🔧 WHAT IT DOES
#    - CheckIn       : one day's entry (date, mood 1-10, hours slept, optional note).
#    - Cipher        : interface for encryption (encrypt / decrypt bytes).
#    - FernetCipher  : authenticated encryption with the `cryptography` package (optional extra).
#    - JournalStore  : load / save / add entries in one encrypted file, written atomically.
#
#  🧠 DESIGN NOTES
#    - Encrypted by default: there is no "plain text" mode in the store. Tests use a fake cipher
#      only through the same interface.
#    - Fernet = AES with an HMAC: a tampered or wrong-key file fails loudly instead of returning
#      garbage. The key is generated once and must be kept by the user (password manager);
#      losing it means the journal cannot be read, by design.
#    - Atomic writes (mattslib.atomic_write): a crash mid-save never corrupts the journal.
#    - Note text is never logged. Repr of a CheckIn hides the note.
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial release: CheckIn, Cipher, FernetCipher, encrypted JournalStore.
# ==================================================================================================
"""Private, encrypted daily check-ins."""

# ========================================
# 📂 Imports
# ========================================
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Protocol

from mattslib import atomic_write


# ========================================
# 📂 Entry
# ========================================
@dataclass(frozen=True, slots=True)
class CheckIn:
    day: date
    mood: int  # 1 (very low) .. 10 (very good)
    sleep_hours: float | None = None
    note: str = field(default="", repr=False)  # never shown in logs or tracebacks

    def __post_init__(self) -> None:
        if not 1 <= self.mood <= 10:
            raise ValueError("mood must be between 1 and 10")
        if self.sleep_hours is not None and not 0 <= self.sleep_hours <= 24:
            raise ValueError("sleep_hours must be between 0 and 24")


# ========================================
# 📂 Encryption
# ========================================
class Cipher(Protocol):
    def encrypt(self, data: bytes) -> bytes: ...

    def decrypt(self, token: bytes) -> bytes: ...


class FernetCipher:
    """Authenticated encryption (AES-128-CBC + HMAC-SHA256) via `cryptography`."""

    def __init__(self, key: bytes) -> None:
        try:
            from cryptography.fernet import Fernet
        except ImportError as exc:
            raise ImportError(
                "encryption needs cryptography: pip install 'ailab-mental-health[crypto]'"
            ) from exc
        self._fernet: Any = Fernet(key)

    @staticmethod
    def generate_key() -> bytes:
        from cryptography.fernet import Fernet

        key: bytes = Fernet.generate_key()
        return key

    def encrypt(self, data: bytes) -> bytes:
        token: bytes = self._fernet.encrypt(data)
        return token

    def decrypt(self, token: bytes) -> bytes:
        try:
            data: bytes = self._fernet.decrypt(token)
        except Exception:  # cryptography raises InvalidToken; do not leak details
            raise ValueError("journal could not be decrypted: wrong key or damaged file") from None
        return data


# ========================================
# 📂 Store
# ========================================
class JournalStore:
    """All check-ins in one encrypted JSON file. One entry per day (later entries replace)."""

    def __init__(self, path: str | Path, cipher: Cipher) -> None:
        self._path = Path(path)
        self._cipher = cipher

    def load(self) -> list[CheckIn]:
        if not self._path.exists():
            return []
        rows = json.loads(self._cipher.decrypt(self._path.read_bytes()))
        return [
            CheckIn(date.fromisoformat(r["day"]), r["mood"], r["sleep_hours"], r["note"])
            for r in rows
        ]

    def save(self, entries: list[CheckIn]) -> None:
        rows = [
            {**asdict(e), "day": e.day.isoformat()} for e in sorted(entries, key=lambda e: e.day)
        ]
        atomic_write(self._path, self._cipher.encrypt(json.dumps(rows).encode()))

    def add(self, entry: CheckIn) -> None:
        entries = {e.day: e for e in self.load()}
        entries[entry.day] = entry
        self.save(list(entries.values()))
