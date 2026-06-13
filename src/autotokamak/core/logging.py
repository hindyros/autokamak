"""Phase-grouped, elapsed-time-stamped console logger for equilibrium runs.

Console output is human-friendly (short MM:SS elapsed stamps, section banners,
aligned key-value lines). The file log keeps full ISO timestamps for forensics.

Critical detail: OFT's compiled code (Fortran / C) writes to libc stdio, which
buffers independently of Python's sys.stdout. Without explicit fflush(NULL),
OFT's banner and iteration output dumps to the terminal after all of Python's
prints, making the log unreadable. _flush_all() forces both streams to flush
together so the order matches what the user expects.
"""

from __future__ import annotations

import ctypes as _ctypes
import datetime as _dt
import sys
from typing import TextIO

# Captured at module import — elapsed time is relative to this.
_RUN_START_TIME = _dt.datetime.utcnow()

try:
    _LIBC = _ctypes.CDLL(None)
except Exception:  # noqa: BLE001
    _LIBC = None


def _flush_all() -> None:
    """Flush Python stdout AND libc stdio so OFT's compiled output stays in order."""
    sys.stdout.flush()
    if _LIBC is not None:
        try:
            _LIBC.fflush(None)  # NULL = flush every open stream
        except Exception:  # noqa: BLE001
            pass


def elapsed() -> str:
    """MM:SS.xx since module load, for compact console stamps."""
    delta = (_dt.datetime.utcnow() - _RUN_START_TIME).total_seconds()
    mm = int(delta // 60)
    ss = delta - mm * 60
    return f"{mm:02d}:{ss:05.2f}"


def log(msg: str, log_fp: TextIO | None = None) -> None:
    """Log one step. Console gets a short elapsed-time stamp; file gets the full ISO timestamp."""
    iso_line = f"[{_dt.datetime.utcnow().isoformat()}Z] {msg}"
    _flush_all()
    print(f"  [{elapsed()}]  {msg}", flush=True)
    if log_fp is not None:
        log_fp.write(iso_line + "\n")
        log_fp.flush()


def section(title: str, log_fp: TextIO | None = None) -> None:
    """Print a section header (console + log) to group related steps visually."""
    bar = "-" * 70
    _flush_all()
    print(f"\n{bar}\n  {title}\n{bar}", flush=True)
    _flush_all()
    if log_fp is not None:
        log_fp.write(f"\n[{_dt.datetime.utcnow().isoformat()}Z] === {title} ===\n")
        log_fp.flush()


def kv(label: str, value, log_fp: TextIO | None = None, *, width: int = 14) -> None:
    """Print a 'label: value' line aligned for readability."""
    print(f"      {label:<{width}} {value}", flush=True)
    if log_fp is not None:
        log_fp.write(f"[{_dt.datetime.utcnow().isoformat()}Z]   {label}: {value}\n")
        log_fp.flush()


__all__ = ["elapsed", "log", "section", "kv"]
