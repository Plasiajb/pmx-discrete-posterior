"""Bounded atomic replacement and short-lived checkpoint reads."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

REPLACE_ATTEMPTS = 21
REPLACE_DELAY_SECONDS = 0.05


def atomic_replace(source, target):
    """Retry Windows access/sharing errors with at most twenty 50 ms waits."""
    for attempt in range(REPLACE_ATTEMPTS):
        try:
            os.replace(source, target)
            return
        except OSError as exc:
            if getattr(exc, 'winerror', None) not in (5, 32, 33) or attempt == REPLACE_ATTEMPTS-1:
                raise
            time.sleep(REPLACE_DELAY_SECONDS)


def write_text_atomic(path, text):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(text, encoding='utf-8')
    atomic_replace(temporary, path)


def read_checkpoint_json(path):
    """Close the short byte-read handle before decoding/parsing the snapshot."""
    raw = Path(path).read_bytes()
    return json.loads(raw.decode('utf-8-sig'))
